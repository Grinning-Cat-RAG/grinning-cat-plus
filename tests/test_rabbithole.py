"""Parser wiring of ``rabbithole_instantiates_parsers``, with the Grinning Cat core stubbed out."""
from __future__ import annotations

import asyncio
import importlib
import sys
import types

import pytest

ODP_MIME = "application/vnd.oasis.opendocument.presentation"
DOC_MIME = "application/msword"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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


class _BaseChunker:
    """Mirror of cat.services.factory.chunker.BaseChunker (lazy ``splitter``)."""

    def __init__(self):
        self._splitter = None

    @property
    def splitter(self):
        if not self._splitter:
            self._splitter = self._get_splitter()
        return self._splitter


class _EmbeddedBaseChunker(_BaseChunker):
    async def _ensure_embedder(self):  # the tests inject a splitter that needs no embedder
        return None


def _stub_cat() -> dict[str, types.ModuleType]:
    cat = types.ModuleType("cat")
    cat.hook = lambda **kwargs: (lambda fn: fn)
    cat.BillTheLizard = _Lizard
    cat.EmbedderSettings = object
    cat.BaseChunker = _BaseChunker
    cat.EmbeddedBaseChunker = _EmbeddedBaseChunker
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


NLTK_DOWNLOADS: list[str] = []


@pytest.fixture(scope="module")
def rabbithole():
    import nltk

    with pytest.MonkeyPatch.context() as mp:
        for name, module in _stub_cat().items():
            mp.setitem(sys.modules, name, module)
        NLTK_DOWNLOADS.clear()
        mp.setattr(nltk, "download", lambda name, **k: NLTK_DOWNLOADS.append(name) or True)  # no network
        module = importlib.import_module("grinning_cat_plus.rabbithole")
        mp.setattr(module.ctranslate2, "get_cuda_device_count", lambda: 0)
        yield module
        sys.modules.pop("grinning_cat_plus.rabbithole", None)


def _handlers(rabbithole, multimodal: bool) -> dict:
    _EmbedderConfig.multimodal = multimodal
    return asyncio.run(rabbithole.rabbithole_instantiates_parsers({}, cat=object()))


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize(
    "mime,source_ext,target_ext,target_mime",
    [(ODP_MIME, ".odp", "pptx", PPTX_MIME), (PPT_MIME, ".ppt", "pptx", PPTX_MIME), (DOC_MIME, ".doc", "docx", DOCX_MIME)],
)
def test_legacy_and_odf_formats_are_converted_then_parsed_like_their_target(
    rabbithole, multimodal, mime, source_ext, target_ext, target_mime
):
    """unstructured would call soffice itself (shared profile, no timeout): we convert first."""
    from grinning_cat_plus.parsers import LibreOfficeConversionParser, UnstructuredParser

    handlers = _handlers(rabbithole, multimodal)
    parser = handlers[mime]
    assert isinstance(parser, LibreOfficeConversionParser)
    assert (parser._source_ext, parser._target_ext) == (source_ext, target_ext)
    assert type(parser._inner) is UnstructuredParser
    assert parser._inner is handlers[target_mime]


@pytest.mark.parametrize("multimodal", [False, True])
def test_ooxml_formats_go_straight_to_unstructured(rabbithole, multimodal):
    from grinning_cat_plus.parsers import UnstructuredParser

    handlers = _handlers(rabbithole, multimodal)
    assert type(handlers[PPTX_MIME]) is UnstructuredParser
    assert type(handlers[DOCX_MIME]) is UnstructuredParser
    expected_strategy = "hi_res" if multimodal else "fast"
    assert handlers[PPTX_MIME]._partition_kwargs["strategy"] == expected_strategy


def test_no_embedder_config_leaves_handlers_untouched(rabbithole, monkeypatch):
    monkeypatch.setattr(_ServiceFactory, "config", None)
    sentinel = {"x/y": object()}
    assert asyncio.run(rabbithole.rabbithole_instantiates_parsers(dict(sentinel), cat=object())) == sentinel


def test_powerpoint_parser_is_not_exported():
    import grinning_cat_plus.parsers as parsers

    assert "PowerPointParser" not in parsers.__all__
    assert "LibreOfficeConversionParser" in parsers.__all__
    assert "OdpParser" not in parsers.__all__


def _load_check_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "check_mimetypes.py"
    spec = importlib.util.spec_from_file_location("check_mimetypes", path)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    return script


@pytest.mark.parametrize("multimodal", [False, True])
def test_check_mimetypes_script_mirrors_the_real_routing(rabbithole, multimodal):
    """scripts/check_mimetypes.py simulates the hook with labels: it must not drift from it."""
    script = _load_check_script()
    real = {mime: type(parser).__name__ for mime, parser in _handlers(rabbithole, multimodal).items()}
    simulated = script._simulate_plugin(rabbithole._get_unstructured_supported_mimetypes(), multimodal)
    assert simulated == real


def test_check_mimetypes_scan_outside_the_cat_matches_the_plugin_scan(rabbithole):
    """Outside the Cat the plugin module is not importable: the inline fallback must agree with it."""
    assert _load_check_script()._get_all_filetypes() == rabbithole._get_unstructured_supported_mimetypes()


def test_check_mimetypes_scan_inside_the_cat_uses_the_plugin_scan(rabbithole, monkeypatch):
    sentinel = {"x/y": object()}
    plugin_module = types.ModuleType("cat.plugins.grinning_cat_plus.rabbithole")
    plugin_module._get_unstructured_supported_mimetypes = lambda: sentinel
    for name in ("cat", "cat.plugins", "cat.plugins.grinning_cat_plus"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "cat.plugins.grinning_cat_plus.rabbithole", plugin_module)
    assert _load_check_script()._get_all_filetypes() is sentinel


EML_MIME = "message/rfc822"
MSG_MIME = "application/vnd.ms-outlook"


def _unstructured_parsers(handlers):
    from grinning_cat_plus.parsers import LibreOfficeConversionParser, UnstructuredParser

    for parser in handlers.values():
        if isinstance(parser, LibreOfficeConversionParser):
            parser = parser._inner
        if isinstance(parser, UnstructuredParser):
            yield parser


@pytest.mark.parametrize("multimodal", [False, True])
def test_no_unstructured_parser_processes_attachments(rabbithole, multimodal):
    """unstructured picks the partitioner from the content, not the MIME label: any parser may
    meet an email and would partition its attachments itself (DOC / PPT: its own soffice,
    shared profile, no timeout). Mailbox ingestion is the email monitoring plugin's job."""
    parsers = list(_unstructured_parsers(_handlers(rabbithole, multimodal)))
    assert len(parsers) > 5
    for parser in parsers:
        assert parser._partition_kwargs["process_attachments"] is False
        assert parser._partition_kwargs["strategy"] == ("hi_res" if multimodal else "fast")


def _eml_with_attachment() -> bytes:
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = "a@x.org", "b@x.org", "Oggetto"
    msg.set_content("BODY_TEXT è qui")
    msg.add_attachment(b"ATTACHMENT_TEXT", maintype="text", subtype="plain", filename="note.txt")
    return msg.as_bytes()


def test_email_content_under_another_label_drops_attachments_too(rabbithole):
    from langchain_core.document_loaders import Blob

    handlers = _handlers(rabbithole, False)
    parser = handlers["text/plain"]
    blob = Blob(data=_eml_with_attachment(), mimetype="text/plain", path="mail.eml")
    text = "\n".join(d.page_content for d in parser.lazy_parse(blob))
    assert "BODY_TEXT" in text
    assert "ATTACHMENT_TEXT" not in text


@pytest.mark.parametrize("fmt", ["txt", "html", "md", "csv", "docx", "pptx"])
def test_the_flag_does_not_break_other_formats(rabbithole, tmp_path, fmt):
    """Every partitioner must accept process_attachments (unstructured forwards **kwargs)."""
    from langchain_core.document_loaders import Blob

    mimes = {
        "txt": "text/plain", "html": "text/html", "md": "text/markdown", "csv": "text/csv",
        "docx": DOCX_MIME, "pptx": PPTX_MIME,
    }
    if fmt == "docx":
        import docx

        d = docx.Document()
        d.add_paragraph("CONTENUTO prova")
        d.save(tmp_path / "f.docx")
        data = (tmp_path / "f.docx").read_bytes()
    elif fmt == "pptx":
        from pptx import Presentation

        prs = Presentation()
        prs.slides.add_slide(prs.slide_layouts[1]).shapes.title.text = "CONTENUTO prova"
        prs.save(tmp_path / "f.pptx")
        data = (tmp_path / "f.pptx").read_bytes()
    else:
        data = {
            "txt": b"CONTENUTO prova\n", "html": b"<html><body><p>CONTENUTO prova</p></body></html>",
            "md": b"# CONTENUTO prova\n", "csv": b"a,b\nCONTENUTO prova,1\n",
        }[fmt]
    parser = _handlers(rabbithole, False)[mimes[fmt]]
    parser = getattr(parser, "_inner", parser)
    text = "\n".join(d.page_content for d in parser.lazy_parse(Blob(data=data, mimetype=mimes[fmt], path=f"f.{fmt}")))
    assert "CONTENUTO prova" in text


def test_uploaded_eml_keeps_the_body_and_drops_attachments(rabbithole):
    from langchain_core.document_loaders import Blob

    blob = Blob(data=_eml_with_attachment(), mimetype=EML_MIME, path="mail.eml")

    parser = _handlers(rabbithole, False)[EML_MIME]
    text = "\n".join(d.page_content for d in parser.lazy_parse(blob))
    assert "BODY_TEXT è qui" in text
    assert "ATTACHMENT_TEXT" not in text

    # Control: the same message with attachment processing on does surface the attachment.
    from grinning_cat_plus.parsers import UnstructuredParser

    control = "\n".join(d.page_content for d in UnstructuredParser(strategy="fast").lazy_parse(blob))
    assert "ATTACHMENT_TEXT" in control


def test_import_downloads_exactly_the_nltk_data_the_plugin_uses(rabbithole):
    """NLTK >= 3.9 sentence tokenization (TokenNLTKChunker) needs `punkt_tab`; nothing in the
    plugin (nor in unstructured 0.22) uses the old `punkt` pickle or the POS tagger."""
    assert NLTK_DOWNLOADS == ["punkt_tab"]
