"""Tokenizer resolution of ``CustomVllmMultimodalEmbedder``, with every dependency stubbed.

Contract pinned here:

* the tokenizer is read from the local HF cache only (``local_files_only=True``), at most once per
  process per model, however many fresh embedder instances are built;
* when the model is not cached, the ``"remote"`` sentinel is sticky and counts come from the vLLM
  ``/tokenize`` endpoint (mocked);
* concurrent fresh instances of the same model share a single load (double-checked locking);
* the first load of one model never blocks the first load of another (per-model locks).

No network is ever reached: ``transformers``, ``httpx`` and ``requests`` are fakes.
"""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path

import pytest

CUSTOM_PATH = Path(__file__).resolve().parent.parent / "embedders" / "custom.py"
MODULE_NAME = "grinning_cat_plus_embedders_custom_under_test"

LOCAL_MODEL = "jinaai/jina-embeddings-v5-omni-small"
REMOTE_MODEL = "jinaai/not-in-local-cache"
SLOW_MODEL = "jinaai/slow-to-load"

# An explicit override makes __init__ skip the /v1/models probe entirely.
MAX_INPUT_TOKENS = 8192

# The fake loader sleeps before recording a call, modelling the file I/O of a real cache read.
# Without it the fake is pure Python, the GIL serializes the threads and the concurrency test
# would pass even with the lock removed; with it, a lock-free implementation loses the race.
FAKE_LOAD_DELAY = 0.05


class _FakeTokenizer:
    """One character == one token, so "abc" is always 3 tokens."""

    def encode(self, text):
        return list(text)


class _FakeAutoTokenizer:
    """Records every ``from_pretrained`` call and never touches the network."""

    calls: list[tuple[str, dict]] = []
    slow_model_entered = threading.Event()
    slow_model_release = threading.Event()

    @classmethod
    def from_pretrained(cls, name, **kwargs):
        if name == SLOW_MODEL:
            cls.slow_model_entered.set()
            assert cls.slow_model_release.wait(timeout=30), "slow model never released"
        else:
            time.sleep(FAKE_LOAD_DELAY)
        cls.calls.append((name, dict(kwargs)))
        if name == REMOTE_MODEL:
            raise OSError(f"{name} is not present in the local HuggingFace cache")
        return _FakeTokenizer()


def _unexpected_network(*args, **kwargs):
    raise AssertionError("unexpected network call: these tests must stay offline")


def _module(name: str, **attributes) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def _stubs() -> dict[str, types.ModuleType]:
    log = _module("cat.log", info=lambda *a, **k: None, warning=lambda *a, **k: None, debug=lambda *a, **k: None)
    utils = _module("cat.utils", retrieve_image=lambda *a, **k: None)
    cat = _module(
        "cat",
        Embeddings=type("Embeddings", (), {}),
        MultimodalEmbeddings=type("MultimodalEmbeddings", (), {}),
        hook=lambda *a, **k: (lambda fn: fn),
        log=log,
        utils=utils,
    )
    cat.__path__ = []
    db = _module("cat.db")
    db.__path__ = []
    cruds = _module("cat.db.cruds")
    cruds.__path__ = []
    return {
        "cat": cat,
        "cat.log": log,
        "cat.utils": utils,
        "cat.db": db,
        "cat.db.database": _module(
            "cat.db.database", DEFAULT_SYSTEM_KEY="system:agent", get_sync_db=_unexpected_network
        ),
        "cat.db.cruds": cruds,
        "cat.db.cruds.settings": _module("cat.db.cruds.settings", format_key=lambda key, *a, **k: key),
        "fastembed": _module(
            "fastembed", TextEmbedding=type("TextEmbedding", (), {"list_supported_models": staticmethod(list)})
        ),
        "sentence_transformers": _module("sentence_transformers", SentenceTransformer=_unexpected_network),
        "transformers": _module("transformers", AutoTokenizer=_FakeAutoTokenizer),
        "httpx": _module("httpx", post=_unexpected_network),
        "requests": _module("requests", post=_unexpected_network, get=_unexpected_network),
    }


@pytest.fixture(scope="module")
def custom():
    with pytest.MonkeyPatch.context() as mp:
        for name, module in _stubs().items():
            mp.setitem(sys.modules, name, module)
        spec = importlib.util.spec_from_file_location(MODULE_NAME, CUSTOM_PATH)
        module = importlib.util.module_from_spec(spec)
        mp.setitem(sys.modules, MODULE_NAME, module)
        spec.loader.exec_module(module)
        yield module


@pytest.fixture(autouse=True)
def _fresh_process_state(custom):
    custom._TOKENIZER_CACHE.clear()
    custom._TOKENIZER_LOCKS.clear()
    _FakeAutoTokenizer.calls.clear()
    _FakeAutoTokenizer.slow_model_entered.clear()
    _FakeAutoTokenizer.slow_model_release.clear()
    yield
    _FakeAutoTokenizer.slow_model_release.set()  # never leave a worker thread blocked


def _build(custom, model):
    return custom.CustomVllmMultimodalEmbedder(
        base_url="http://127.0.0.1:9", model=model, max_input_tokens=MAX_INPUT_TOKENS
    )


def _calls_for(model):
    return [kwargs for name, kwargs in _FakeAutoTokenizer.calls if name == model]


def _run_threads(target, count):
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "worker threads did not finish: possible deadlock"


def test_local_tokenizer_is_loaded_once_per_process_and_offline(custom):
    instances = [_build(custom, LOCAL_MODEL) for _ in range(3)]

    assert [instance._estimate_text_tokens("abc") for instance in instances] == [3, 3, 3]
    assert _calls_for(LOCAL_MODEL) == [{"trust_remote_code": True, "local_files_only": True}]


def test_remote_sentinel_is_sticky_and_counts_come_from_vllm_tokenize(custom, monkeypatch):
    posted = []

    class _Response:
        status_code = 200

        @staticmethod
        def json():
            return {"count": 7}

    def fake_post(url, **kwargs):
        posted.append(url)
        return _Response()

    monkeypatch.setattr(custom.httpx, "post", fake_post)

    counts = [_build(custom, REMOTE_MODEL)._estimate_text_tokens(text) for text in ("x" * 100, "y" * 100)]

    assert counts == [7, 7]
    assert posted == ["http://127.0.0.1:9/tokenize"] * 2
    assert len(_calls_for(REMOTE_MODEL)) == 1, "the 'remote' sentinel must stop further load attempts"
    assert custom._TOKENIZER_CACHE[REMOTE_MODEL] == "remote"


def test_concurrent_fresh_instances_share_a_single_load(custom):
    thread_count = 8
    barrier = threading.Barrier(thread_count)
    counts, errors = [], []
    guard = threading.Lock()

    def worker():
        try:
            instance = _build(custom, LOCAL_MODEL)
            barrier.wait(timeout=30)
            value = instance._estimate_text_tokens("abc")
            with guard:
                counts.append(value)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            with guard:
                errors.append(exc)

    _run_threads(worker, thread_count)

    assert not errors, f"worker threads raised: {errors!r}"
    assert counts == [3] * thread_count
    assert len(_calls_for(LOCAL_MODEL)) == 1


def test_a_slow_load_of_one_model_does_not_block_another_model(custom):
    slow = threading.Thread(target=lambda: _build(custom, SLOW_MODEL)._estimate_text_tokens("abc"))
    slow.start()
    try:
        assert _FakeAutoTokenizer.slow_model_entered.wait(timeout=30), "slow load never started"

        other = threading.Thread(target=lambda: _build(custom, LOCAL_MODEL)._estimate_text_tokens("abc"))
        other.start()
        other.join(timeout=5)

        assert not other.is_alive(), "loading another model waited on the slow model's lock"
        assert len(_calls_for(LOCAL_MODEL)) == 1
    finally:
        _FakeAutoTokenizer.slow_model_release.set()
        slow.join(timeout=30)
    assert len(_calls_for(SLOW_MODEL)) == 1
