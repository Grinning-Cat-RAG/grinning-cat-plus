from __future__ import annotations

from typing import Iterator

import pytest
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document

from grinning_cat_plus.parsers.conversion_parser import (
    DOCX_MIME,
    PPTX_MIME,
    LibreOfficeConversionParser,
)

ODP_MIME = "application/vnd.oasis.opendocument.presentation"
PPT_MIME = "application/vnd.ms-powerpoint"
DOC_MIME = "application/msword"

# (source mime, source extension, target extension, expected target mime)
FORMATS = [
    pytest.param(ODP_MIME, ".odp", "pptx", PPTX_MIME, id="odp"),
    pytest.param(PPT_MIME, ".ppt", "pptx", PPTX_MIME, id="ppt"),
    pytest.param(DOC_MIME, ".doc", "docx", DOCX_MIME, id="doc"),
]


class RecordingParser(BaseBlobParser):
    """Inner parser double: records the blobs it receives, yields ``n_docs`` documents."""

    def __init__(self, n_docs: int = 2, fail_after: int | None = None) -> None:
        self.n_docs = n_docs
        self.fail_after = fail_after
        self.blobs: list[Blob] = []

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        self.blobs.append(blob)
        for i in range(self.n_docs):
            if self.fail_after is not None and i == self.fail_after:
                raise ValueError("inner parser failed")
            yield Document(page_content=f"doc{i}", metadata={"filename": blob.source})


def _blob(mime: str, source: str | None, data: bytes = b"BYTES") -> Blob:
    return Blob.from_data(data=data, mime_type=mime, path=source)


def _parser(inner, source_ext=".odp", target_ext="pptx", **kwargs):
    return LibreOfficeConversionParser(inner, source_ext=source_ext, target_ext=target_ext, **kwargs)


@pytest.mark.parametrize("mime,source_ext,target_ext,target_mime", FORMATS)
def test_converts_and_delegates_with_original_source(
    fake_soffice, isolated_tmp, mime, source_ext, target_ext, target_mime
):
    inner = RecordingParser()
    source = f"Documento è già{source_ext}"
    docs = list(_parser(inner, source_ext, target_ext).lazy_parse(_blob(mime, source)))
    assert [d.page_content for d in docs] == ["doc0", "doc1"]
    [received] = inner.blobs
    assert received.mimetype == target_mime
    assert received.as_bytes() == b"CONVERTED:BYTES"
    assert received.source == source
    [call] = fake_soffice.calls()
    assert f"--convert-to {target_ext} " in call
    # Fixed ASCII name: the user-supplied source (non-ASCII, spaces, URLs) never reaches soffice.
    assert call.endswith(f"/input{source_ext}")
    assert list(isolated_tmp.iterdir()) == []


def test_url_source_never_reaches_the_command_line(fake_soffice, isolated_tmp):
    list(_parser(RecordingParser()).lazy_parse(_blob(ODP_MIME, "https://x.org/dl?id=1&n=è")))
    [call] = fake_soffice.calls()
    assert call.endswith("/input.odp")
    assert "x.org" not in call


def test_unsupported_target_is_rejected_at_construction():
    with pytest.raises(ValueError, match="xlsx"):
        _parser(RecordingParser(), ".ods", "xlsx")


def test_blob_metadata_is_preserved(fake_soffice, isolated_tmp):
    inner = RecordingParser()
    blob = Blob(data=b"X", mimetype=ODP_MIME, path="a.odp", metadata={"tenant": "t1"})
    list(_parser(inner).lazy_parse(blob))
    assert inner.blobs[0].metadata == {"tenant": "t1"}


def test_blob_without_source(fake_soffice, isolated_tmp):
    inner = RecordingParser()
    list(_parser(inner).lazy_parse(_blob(ODP_MIME, None)))
    assert inner.blobs[0].source is None


def test_errors_name_the_original_source(fake_soffice, isolated_tmp):
    fake_soffice.mode("fail")
    with pytest.raises(RuntimeError, match="Presentazione è già.odp"):
        list(_parser(RecordingParser()).lazy_parse(_blob(ODP_MIME, "Presentazione è già.odp")))


@pytest.mark.parametrize(
    "mime,source_ext,target_ext,label",
    [(ODP_MIME, ".odp", "pptx", "ODP file"), (DOC_MIME, ".doc", "docx", "DOC file")],
)
def test_errors_without_source_name_the_format(fake_soffice, isolated_tmp, mime, source_ext, target_ext, label):
    fake_soffice.mode("fail")
    with pytest.raises(RuntimeError, match=label):
        list(_parser(RecordingParser(), source_ext, target_ext).lazy_parse(_blob(mime, None)))


def test_conversion_failure_propagates_and_cleans_up(fake_soffice, isolated_tmp):
    fake_soffice.mode("fail")
    inner = RecordingParser()
    with pytest.raises(RuntimeError, match="exit 3"):
        list(_parser(inner).lazy_parse(_blob(ODP_MIME, "a.odp")))
    assert inner.blobs == []
    assert list(isolated_tmp.iterdir()) == []


def test_timeout_is_forwarded(fake_soffice, isolated_tmp):
    fake_soffice.mode("hang")
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        list(_parser(RecordingParser(), timeout=1).lazy_parse(_blob(ODP_MIME, "a.odp")))
    assert list(isolated_tmp.iterdir()) == []


def test_workdir_is_gone_before_the_first_document(fake_soffice, isolated_tmp):
    """A consumer that stops iterating must not keep the conversion directory alive."""
    gen = _parser(RecordingParser()).lazy_parse(_blob(ODP_MIME, "a.odp"))
    next(gen)
    assert list(isolated_tmp.iterdir()) == []
    gen.close()


def test_inner_failure_propagates(fake_soffice, isolated_tmp):
    with pytest.raises(ValueError, match="inner parser failed"):
        list(_parser(RecordingParser(fail_after=1)).lazy_parse(_blob(ODP_MIME, "a.odp")))
    assert list(isolated_tmp.iterdir()) == []


def test_default_timeout_is_the_converter_default():
    from grinning_cat_plus.parsers import libreoffice

    assert _parser(RecordingParser())._timeout == libreoffice.DEFAULT_TIMEOUT_S


class FiletypeParser(BaseBlobParser):
    """Inner double that, like unstructured, stamps the partitioned format into ``filetype``."""

    def __init__(self, stamp: bool = True) -> None:
        self.stamp = stamp
        self.blobs: list[Blob] = []

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        self.blobs.append(blob)
        yield Document(page_content="x", metadata={"filetype": blob.mimetype} if self.stamp else {"page": 1})


@pytest.mark.parametrize("mime,source_ext,target_ext,target_mime", FORMATS)
def test_converted_documents_keep_the_original_filetype(
    fake_soffice, isolated_tmp, mime, source_ext, target_ext, target_mime
):
    """Stored chunk metadata must say what the user uploaded (as partition_doc/ppt did), not the
    intermediate format."""
    inner = FiletypeParser()
    [doc] = _parser(inner, source_ext, target_ext).lazy_parse(_blob(mime, f"f{source_ext}"))
    assert inner.blobs[0].mimetype == target_mime
    assert doc.metadata["filetype"] == mime


def test_filetype_is_not_invented_when_the_inner_parser_sets_none(fake_soffice, isolated_tmp):
    [doc] = _parser(FiletypeParser(stamp=False)).lazy_parse(_blob(ODP_MIME, "f.odp"))
    assert doc.metadata == {"page": 1}


@pytest.mark.parametrize(
    "mime,source_ext,target_ext,target_mime",
    [(DOC_MIME, ".doc", "docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
     (PPT_MIME, ".ppt", "pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation")],
)
def test_ooxml_bytes_under_a_legacy_mime_skip_libreoffice(
    fake_soffice, isolated_tmp, monkeypatch, mime, source_ext, target_ext, target_mime
):
    """Servers often label .docx/.pptx as msword/ms-powerpoint: those bytes are already the
    target format, so no conversion (and no LibreOffice) is needed."""
    monkeypatch.setenv("PATH", str(isolated_tmp))  # no soffice at all
    inner = FiletypeParser()
    data = b"PK\x03\x04 rest of the zip"
    [doc] = _parser(inner, source_ext, target_ext).lazy_parse(_blob(mime, "https://h/dl?id=1", data))
    [received] = inner.blobs
    assert received.mimetype == target_mime
    assert received.as_bytes() == data
    assert received.source == "https://h/dl?id=1"
    assert doc.metadata["filetype"] == target_mime  # the bytes really are OOXML
    assert fake_soffice.calls() == []


def test_odp_is_zip_too_and_is_still_converted(fake_soffice, isolated_tmp):
    inner = RecordingParser(n_docs=1)
    list(_parser(inner).lazy_parse(_blob(ODP_MIME, "f.odp", b"PK\x03\x04odp")))
    assert inner.blobs[0].as_bytes() == b"CONVERTED:PK\x03\x04odp"
    assert len(fake_soffice.calls()) == 1
