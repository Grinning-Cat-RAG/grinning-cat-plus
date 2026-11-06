"""Parser wiring of ``rabbithole_instantiates_parsers``, with the Grinning Cat core stubbed out."""
from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

ODP_MIME = "application/vnd.oasis.opendocument.presentation"
PPT_MIME = "application/vnd.ms-powerpoint"
PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class _EmbedderConfig:
    multimodal = False

    @classmethod
    def is_multimodal(cls):
        return cls.multimodal


class _ServiceFactory:
    config: type | None = _EmbedderConfig

    def __init__(self, **kwargs):
        pass

    async def get_config_class_from_adapter(self, adapter):
        return _ServiceFactory.config


class _Lizard:
    agent_key = "agent"
    plugin_manager = None

    async def embedder(self):
        return object()


def _stub_cat() -> dict[str, types.ModuleType]:
    cat = types.ModuleType("cat")
    cat.hook = lambda **kwargs: (lambda fn: fn)
    cat.BillTheLizard = _Lizard
    cat.EmbedderSettings = object
    cat.BaseChunker = type("BaseChunker", (), {})
    cat.EmbeddedBaseChunker = type("EmbeddedBaseChunker", (), {})
    cat.Embeddings = object
    services = types.ModuleType("cat.services")
    service_factory = types.ModuleType("cat.services.service_factory")
    service_factory.ServiceFactory = _ServiceFactory
    factory = types.ModuleType("cat.services.factory")
    chunker = types.ModuleType("cat.services.factory.chunker")
    chunker.ChunkerSettings = object
    return {
        "cat": cat,
        "cat.services": services,
        "cat.services.service_factory": service_factory,
        "cat.services.factory": factory,
        "cat.services.factory.chunker": chunker,
    }


@pytest.fixture(scope="module")
def rabbithole():
    import nltk

    with pytest.MonkeyPatch.context() as mp:
        for name, module in _stub_cat().items():
            mp.setitem(sys.modules, name, module)
        mp.setattr(nltk, "download", lambda *a, **k: True)  # no network at import time
        module = importlib.import_module("grinning_cat_plus.rabbithole")
        mp.setattr(module.ctranslate2, "get_cuda_device_count", lambda: 0)
        yield module
        sys.modules.pop("grinning_cat_plus.rabbithole", None)


def _handlers(rabbithole, multimodal: bool) -> dict:
    _EmbedderConfig.multimodal = multimodal
    return asyncio.run(rabbithole.rabbithole_instantiates_parsers({}, cat=object()))


@pytest.mark.parametrize("multimodal", [False, True])
def test_odp_is_converted_then_parsed_like_pptx(rabbithole, multimodal):
    from grinning_cat_plus.parsers import OdpParser, UnstructuredParser

    handlers = _handlers(rabbithole, multimodal)
    odp = handlers[ODP_MIME]
    assert isinstance(odp, OdpParser)
    assert isinstance(odp._inner, UnstructuredParser)
    assert odp._inner._partition_kwargs == handlers[PPTX_MIME]._partition_kwargs


@pytest.mark.parametrize("multimodal", [False, True])
def test_ppt_and_pptx_stay_on_unstructured(rabbithole, multimodal):
    from grinning_cat_plus.parsers import UnstructuredParser

    handlers = _handlers(rabbithole, multimodal)
    assert type(handlers[PPT_MIME]) is UnstructuredParser
    assert type(handlers[PPTX_MIME]) is UnstructuredParser
    expected_strategy = "hi_res" if multimodal else "fast"
    assert handlers[PPTX_MIME]._partition_kwargs["strategy"] == expected_strategy


def test_no_embedder_config_leaves_handlers_untouched(rabbithole, monkeypatch):
    monkeypatch.setattr(_ServiceFactory, "config", None)
    sentinel = {"x/y": object()}
    assert asyncio.run(rabbithole.rabbithole_instantiates_parsers(dict(sentinel), cat=object())) == sentinel


def test_powerpoint_parser_is_not_exported():
    import grinning_cat_plus.parsers as parsers

    assert "PowerPointParser" not in parsers.__all__
    assert "OdpParser" in parsers.__all__
