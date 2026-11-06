from __future__ import annotations

from typing import Iterator

import pytest
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document

from grinning_cat_plus.parsers.odp_parser import OdpParser, PPTX_MIME

ODP_MIME = "application/vnd.oasis.opendocument.presentation"


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


def _odp_blob(source: str | None = "Presentazione è già.odp", data: bytes = b"ODP-BYTES") -> Blob:
    return Blob.from_data(data=data, mime_type=ODP_MIME, path=source)


def test_converts_to_pptx_and_delegates_with_original_source(fake_soffice, isolated_tmp):
    inner = RecordingParser()
    docs = list(OdpParser(inner).lazy_parse(_odp_blob()))
    assert [d.page_content for d in docs] == ["doc0", "doc1"]
    [received] = inner.blobs
    assert received.mimetype == PPTX_MIME
    assert received.as_bytes() == b"CONVERTED:ODP-BYTES"
    assert received.source == "Presentazione è già.odp"
    assert list(isolated_tmp.iterdir()) == []


def test_internal_file_name_is_fixed_and_ascii(fake_soffice, isolated_tmp):
    """The user-supplied name never reaches the command line (non-ASCII, spaces, URLs)."""
    list(OdpParser(RecordingParser()).lazy_parse(_odp_blob("https://x.org/dl?id=1&n=è")))
    [call] = fake_soffice.calls()
    assert call.endswith("/input.odp")


def test_blob_metadata_is_preserved(fake_soffice, isolated_tmp):
    inner = RecordingParser()
    blob = Blob(data=b"X", mimetype=ODP_MIME, path="a.odp", metadata={"tenant": "t1"})
    list(OdpParser(inner).lazy_parse(blob))
    assert inner.blobs[0].metadata == {"tenant": "t1"}


def test_blob_without_source(fake_soffice, isolated_tmp):
    inner = RecordingParser()
    list(OdpParser(inner).lazy_parse(_odp_blob(source=None)))
    assert inner.blobs[0].source is None


def test_conversion_failure_propagates_and_cleans_up(fake_soffice, isolated_tmp):
    fake_soffice.mode("fail")
    inner = RecordingParser()
    with pytest.raises(RuntimeError, match="exit 3"):
        list(OdpParser(inner).lazy_parse(_odp_blob()))
    assert inner.blobs == []
    assert list(isolated_tmp.iterdir()) == []


def test_timeout_is_forwarded(fake_soffice, isolated_tmp):
    fake_soffice.mode("hang")
    with pytest.raises(RuntimeError, match="timed out after 1s"):
        list(OdpParser(RecordingParser(), timeout=1).lazy_parse(_odp_blob()))
    assert list(isolated_tmp.iterdir()) == []


def test_workdir_is_gone_before_the_first_document(fake_soffice, isolated_tmp):
    """A consumer that stops iterating must not keep the conversion directory alive."""
    gen = OdpParser(RecordingParser()).lazy_parse(_odp_blob())
    next(gen)
    assert list(isolated_tmp.iterdir()) == []
    gen.close()


def test_inner_failure_propagates(fake_soffice, isolated_tmp):
    with pytest.raises(ValueError, match="inner parser failed"):
        list(OdpParser(RecordingParser(fail_after=1)).lazy_parse(_odp_blob()))
    assert list(isolated_tmp.iterdir()) == []


def test_default_timeout_is_the_converter_default():
    from grinning_cat_plus.parsers import libreoffice

    assert OdpParser(RecordingParser())._timeout == libreoffice.DEFAULT_TIMEOUT_S
