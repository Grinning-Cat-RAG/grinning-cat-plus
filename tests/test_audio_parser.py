"""``FasterWhisperParser`` loads its Whisper model lazily, offline-first, once per process.

``faster_whisper.WhisperModel`` is replaced by a fake that records its calls: no model is
downloaded or loaded, and no network is reached.
"""
from __future__ import annotations

import threading
import time
import types

import pytest
from langchain_core.document_loaders import Blob

from grinning_cat_plus.parsers import custom_audio_parser
from grinning_cat_plus.parsers.custom_audio_parser import FasterWhisperParser


class _NotCached(Exception):
    """Stands in for huggingface_hub's LocalEntryNotFoundError."""


class _FakeWhisperModel:
    """Records constructor calls; ``cached`` and ``download_fails`` script the outcome."""

    calls: list[dict] = []
    cached = True
    download_fails = False

    def __init__(self, model_size, **kwargs):
        time.sleep(0.02)  # a real load takes time: lets unsynchronized threads overlap
        _FakeWhisperModel.calls.append({"model_size": model_size, **kwargs})
        if kwargs.get("local_files_only") and not _FakeWhisperModel.cached:
            raise _NotCached(f"{model_size} is not in the local cache")
        if not kwargs.get("local_files_only") and _FakeWhisperModel.download_fails:
            raise ConnectionError("huggingface.co unreachable")
        self.model_size = model_size

    def transcribe(self, audio_file, beam_size):
        return [types.SimpleNamespace(text="hello"), types.SimpleNamespace(text="world")], None


@pytest.fixture(autouse=True)
def fake_whisper(monkeypatch):
    monkeypatch.setattr(custom_audio_parser, "WhisperModel", _FakeWhisperModel)
    monkeypatch.setattr(FasterWhisperParser, "_models", {})
    _FakeWhisperModel.calls = []
    _FakeWhisperModel.cached = True
    _FakeWhisperModel.download_fails = False


def _transcribe(parser):
    return list(parser.lazy_parse(Blob.from_data(b"fake audio", path="clip.wav")))


def test_building_parsers_loads_no_model():
    """The hook builds one parser per audio MIME type on every call: that must cost nothing."""
    [FasterWhisperParser(device="cpu") for _ in range(7)]

    assert _FakeWhisperModel.calls == []


def test_the_model_is_loaded_once_offline_and_shared_by_every_parser():
    parsers = [FasterWhisperParser(device="cpu") for _ in range(7)]

    documents = [doc for parser in parsers for doc in _transcribe(parser)]

    assert [doc.page_content for doc in documents] == ["hello world"] * 7
    assert documents[0].metadata == {"source": "clip.wav"}
    assert _FakeWhisperModel.calls == [
        {"model_size": "base", "device": "cpu", "compute_type": "int8", "local_files_only": True}
    ]


def test_a_model_missing_from_the_cache_is_downloaded_once():
    _FakeWhisperModel.cached = False

    _transcribe(FasterWhisperParser())
    _transcribe(FasterWhisperParser())

    assert [call.get("local_files_only") for call in _FakeWhisperModel.calls] == [True, None]


def test_a_failed_download_is_not_memoized():
    _FakeWhisperModel.cached = False
    _FakeWhisperModel.download_fails = True
    parser = FasterWhisperParser()

    with pytest.raises(ConnectionError):
        _transcribe(parser)

    _FakeWhisperModel.download_fails = False
    assert [doc.page_content for doc in _transcribe(parser)] == ["hello world"]
    assert [call.get("local_files_only") for call in _FakeWhisperModel.calls] == [True, None, True, None]


def test_concurrent_first_transcriptions_share_a_single_load():
    thread_count = 8
    barrier = threading.Barrier(thread_count)
    results, errors = [], []

    def worker():
        try:
            parser = FasterWhisperParser()
            barrier.wait(timeout=30)
            results.append(_transcribe(parser)[0].page_content)
        except BaseException as exc:  # noqa: BLE001 - surfaced by the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not errors, errors
    assert results == ["hello world"] * thread_count
    assert len(_FakeWhisperModel.calls) == 1


def test_models_are_keyed_by_size_device_and_compute_type():
    for parser in (
        FasterWhisperParser(device="cpu"),
        FasterWhisperParser(device="cuda"),
        FasterWhisperParser(device="cpu", model_size="small"),
        FasterWhisperParser(device="cpu", compute_type="float32"),
        FasterWhisperParser(device="cpu"),
    ):
        _transcribe(parser)

    assert [(c["model_size"], c["device"], c["compute_type"]) for c in _FakeWhisperModel.calls] == [
        ("base", "cpu", "int8"),
        ("base", "cuda", "int8"),
        ("small", "cpu", "int8"),
        ("base", "cpu", "float32"),
    ]
