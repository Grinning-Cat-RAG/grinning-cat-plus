#!/usr/bin/env python3
"""Guard: the vLLM multimodal embedder stays offline and loads its tokenizer
at most once per process per model.

Contract pinned here, one assertion per concern:

  a) three fresh ``CustomVllmMultimodalEmbedder`` instances for the same model
     each count "abc" as 3 tokens, share EXACTLY ONE
     ``AutoTokenizer.from_pretrained`` call, and that call passes
     ``local_files_only=True`` -> no anonymous HuggingFace Hub request;
  b) when the model is absent from the local HF cache, ``from_pretrained``
     raises ``OSError`` and the embedder falls back to the vLLM ``/tokenize``
     endpoint (HTTP mocked); the "remote" sentinel is sticky per process, so a
     second fresh instance returns 7 with NO second ``from_pretrained`` call;
  c) eight threads, each with its own fresh instance of the same model, trigger
     exactly ONE further ``from_pretrained`` call for that model.

Run as a plain script::

    cd /root/git/plugins/grinning-cat-plus && python tests/check_offline_tokenizer.py

Prints ``OK`` and exits 0 when everything holds; raises ``AssertionError``
naming the violated expectation otherwise. Importing this module executes
nothing and requires no prerequisites: the CAT host imports every ``.py`` file
under a plugin folder, tests included, so the entire body lives under
``if __name__ == "__main__":``.
"""

if __name__ == "__main__":
    import importlib
    import importlib.util
    import sys
    import threading
    import types
    from pathlib import Path

    sys.dont_write_bytecode = True  # leave no __pycache__ behind in the repo

    PLUGIN_ROOT = Path(__file__).resolve().parent.parent
    CUSTOM_PATH = PLUGIN_ROOT / "embedders" / "custom.py"

    LOCAL_MODEL = "jinaai/jina-embeddings-v5-omni-small"
    REMOTE_MODEL = "jinaai/not-in-local-cache"
    CONCURRENT_MODEL = "jinaai/concurrent-fresh-instances-model"

    # An explicit override makes __init__ skip the /v1/models probe entirely.
    MAX_INPUT_TOKENS = 8192

    # (model_name, kwargs) appended by the fake AutoTokenizer.from_pretrained.
    TOKENIZER_CALLS: list[tuple[str, dict[str, object]]] = []

    class QuietLogger:
        """Stands in for the CAT log module: no output, no side effects."""

        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    class FakeTokenizer:
        """One character == one token, so "abc" is always 3 tokens."""

        def encode(self, text):
            return list(text)

    class FakeAutoTokenizer:
        """Records every from_pretrained call and never touches the network."""

        @staticmethod
        def from_pretrained(name, **kwargs):
            TOKENIZER_CALLS.append((name, dict(kwargs)))
            if name == REMOTE_MODEL:
                raise OSError(f"{name} is not present in the local HuggingFace cache")
            return FakeTokenizer()

    def make_module(name, **attributes):
        module = types.ModuleType(name)
        for attribute, value in attributes.items():
            setattr(module, attribute, value)
        return module

    def unexpected_network(*args, **kwargs):
        raise AssertionError("unexpected network call: the guard must stay offline")

    def install_stubs():
        """Install every dependency of custom.py BEFORE importing it."""
        cat = make_module(
            "cat",
            Embeddings=type("Embeddings", (), {}),
            MultimodalEmbeddings=type("MultimodalEmbeddings", (), {}),
            hook=lambda *args, **kwargs: (lambda fn: fn),
        )
        cat.__path__ = []  # a package, so `from cat.db... import ...` resolves

        logger = QuietLogger()
        cat_log = make_module(
            "cat.log",
            info=logger.info,
            warning=logger.warning,
            debug=logger.debug,
        )
        cat_utils = make_module("cat.utils", retrieve_image=lambda *a, **k: None)
        setattr(cat, "utils", cat_utils)

        cat_db = make_module("cat.db")
        cat_db.__path__ = []
        cat_db_cruds = make_module("cat.db.cruds")
        cat_db_cruds.__path__ = []
        database = make_module(
            "cat.db.database",
            DEFAULT_SYSTEM_KEY="system:agent",
            get_sync_db=unexpected_network,
        )
        settings = make_module(
            "cat.db.cruds.settings",
            format_key=lambda key, *args, **kwargs: key,
        )

        class FakeTextEmbedding:
            @staticmethod
            def list_supported_models():
                return []

        class FakeSentenceTransformer:
            def __init__(self, *args, **kwargs):
                raise AssertionError("SentenceTransformer must not be instantiated")

        stubs = {
            "cat": cat,
            "cat.log": cat_log,
            "cat.utils": cat_utils,
            "cat.db": cat_db,
            "cat.db.database": database,
            "cat.db.cruds": cat_db_cruds,
            "cat.db.cruds.settings": settings,
            "fastembed": make_module("fastembed", TextEmbedding=FakeTextEmbedding),
            "sentence_transformers": make_module(
                "sentence_transformers", SentenceTransformer=FakeSentenceTransformer
            ),
            # Never import the real transformers: the fake is the network guard.
            "transformers": make_module("transformers", AutoTokenizer=FakeAutoTokenizer),
        }

        for optional in ("httpx", "requests"):
            try:
                importlib.import_module(optional)
            except ImportError:  # keep custom.py importable in a bare environment
                stubs[optional] = make_module(optional, post=unexpected_network)

        sys.modules.update(stubs)

    def import_under_test():
        spec = importlib.util.spec_from_file_location("plus_custom_under_test", CUSTOM_PATH)
        assert spec is not None and spec.loader is not None, f"cannot load {CUSTOM_PATH}"
        module = importlib.util.module_from_spec(spec)
        sys.modules["plus_custom_under_test"] = module
        spec.loader.exec_module(module)
        return module

    def build(custom, model):
        return custom.CustomVllmMultimodalEmbedder(
            base_url="http://127.0.0.1:9",
            model=model,
            max_input_tokens=MAX_INPUT_TOKENS,
        )

    def calls_for(model):
        return [kwargs for name, kwargs in TOKENIZER_CALLS if name == model]

    def check_local_cache_loaded_once(custom):
        instances = [build(custom, LOCAL_MODEL) for _ in range(3)]
        tokens = [instance._estimate_text_tokens("abc") for instance in instances]
        assert tokens == [3, 3, 3], (
            "expected each of 3 fresh instances to count 'abc' as 3 tokens with the "
            f"fake tokenizer; actual token counts={tokens}"
        )

        observed = calls_for(LOCAL_MODEL)
        flags = [kwargs.get("local_files_only") for kwargs in observed]
        assert len(observed) == 1 and flags == [True], (
            f"expected exactly 1 AutoTokenizer.from_pretrained call for {LOCAL_MODEL} "
            "with local_files_only=True across 3 fresh instances (no HuggingFace Hub "
            f"contact); actual calls={len(observed)}, local_files_only flags={flags}"
        )

    def check_remote_sentinel_memoized(custom):
        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"count": 7}

        class FakeHttpx:
            @staticmethod
            def post(*args, **kwargs):
                return FakeResponse()

        original_httpx = custom.httpx
        try:
            custom.httpx = FakeHttpx()
            first = build(custom, REMOTE_MODEL)
            first_count = first._estimate_text_tokens("x" * 100)
            second = build(custom, REMOTE_MODEL)
            second_count = second._estimate_text_tokens("y" * 100)
        finally:
            custom.httpx = original_httpx

        assert (first_count, second_count) == (7, 7), (
            "expected both fresh instances to return the mocked vLLM /tokenize count 7 "
            f"when the model is missing from the local cache; actual counts="
            f"{(first_count, second_count)}"
        )

        observed = calls_for(REMOTE_MODEL)
        assert len(observed) == 1, (
            f"expected exactly 1 AutoTokenizer.from_pretrained call for {REMOTE_MODEL} "
            "because the 'remote' sentinel is sticky per process; "
            f"actual calls={len(observed)}"
        )

    def check_concurrent_instances_load_once(custom):
        thread_count = 8
        barrier = threading.Barrier(thread_count)
        counts = []
        errors = []
        guard = threading.Lock()

        def worker():
            try:
                instance = build(custom, CONCURRENT_MODEL)
                barrier.wait(timeout=30)
                value = instance._estimate_text_tokens("abc")
                with guard:
                    counts.append(value)
            except BaseException as exc:  # noqa: BLE001 - surfaced as an assertion
                with guard:
                    errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(thread_count)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=60)

        assert not any(thread.is_alive() for thread in threads), (
            "worker threads did not finish: possible deadlock in the tokenizer cache"
        )
        assert not errors, f"worker threads raised: {errors!r}"
        assert counts == [3] * thread_count, (
            f"expected all {thread_count} concurrent instances to count 'abc' as 3 "
            f"tokens; actual token counts={sorted(counts)}"
        )

        observed = calls_for(CONCURRENT_MODEL)
        assert len(observed) == 1, (
            f"expected exactly 1 further AutoTokenizer.from_pretrained call for "
            f"{CONCURRENT_MODEL} across {thread_count} concurrent fresh instances; "
            f"actual calls={len(observed)}"
        )

    install_stubs()
    custom = import_under_test()

    tokenizer_cache = getattr(custom, "_TOKENIZER_CACHE", None)  # absent today
    if tokenizer_cache is not None:
        tokenizer_cache.clear()

    check_local_cache_loaded_once(custom)
    check_remote_sentinel_memoized(custom)
    check_concurrent_instances_load_once(custom)

    print("OK")
