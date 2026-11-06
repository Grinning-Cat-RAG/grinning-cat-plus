from __future__ import annotations

import os

import pytest
from langchain_core.document_loaders import Blob

from grinning_cat_plus.parsers.unstructured_parser import UnstructuredParser

PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class _Meta:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return self._d


class _Element:
    def __init__(self, text, meta):
        self.text, self.metadata = text, meta

    def __str__(self):
        return self.text


@pytest.fixture
def spy_partition(monkeypatch):
    import unstructured.partition.auto as auto

    calls = []

    def fake_partition(filename, **kwargs):
        calls.append({"filename": filename, "exists": os.path.exists(filename), **kwargs})
        return [_Element("hello", _Meta({"page_number": 1})), _Element("raw", object())]

    monkeypatch.setattr(auto, "partition", fake_partition)
    return calls


def test_mimetype_wins_over_source_extension(spy_partition, isolated_tmp):
    """An ODP converted to PPTX keeps its original ``.odp`` source but must be partitioned as PPTX."""
    blob = Blob(data=b"PK", mimetype=PPTX_MIME, path="deck.odp")
    docs = list(UnstructuredParser(strategy="fast").lazy_parse(blob))
    [call] = spy_partition
    assert call["content_type"] == PPTX_MIME
    assert call["filename"].endswith(".pptx")
    assert call["exists"]
    assert call["metadata_filename"] == "deck.odp"
    assert call["strategy"] == "fast"
    assert [d.page_content for d in docs] == ["hello", "raw"]
    assert docs[0].metadata == {"page_number": 1}
    assert docs[1].metadata == {}
    assert list(isolated_tmp.iterdir()) == []


def test_url_source_without_extension_uses_mimetype(spy_partition, isolated_tmp):
    blob = Blob(data=b"PK", mimetype=DOCX_MIME, path="https://host/download?id=3")
    list(UnstructuredParser().lazy_parse(blob))
    assert spy_partition[0]["content_type"] == DOCX_MIME
    assert spy_partition[0]["filename"].endswith(".docx")


def test_unknown_mimetype_falls_back_to_source_extension(spy_partition, isolated_tmp):
    blob = Blob(data=b"PK", mimetype="application/octet-stream", path="Report.DOCX")
    list(UnstructuredParser().lazy_parse(blob))
    assert spy_partition[0]["content_type"] == DOCX_MIME
    assert spy_partition[0]["filename"].endswith(".docx")


def test_no_hint_at_all_lets_unstructured_detect(spy_partition, isolated_tmp):
    blob = Blob(data=b"%PDF", mimetype="application/pdf", path=None)
    list(UnstructuredParser().lazy_parse(blob))
    assert "content_type" not in spy_partition[0]
    assert spy_partition[0]["metadata_filename"] == spy_partition[0]["filename"]


def test_temp_file_removed_when_partition_fails(monkeypatch, isolated_tmp):
    import unstructured.partition.auto as auto

    def boom(filename, **kwargs):
        raise ValueError("bad file")

    monkeypatch.setattr(auto, "partition", boom)
    with pytest.raises(ValueError, match="bad file"):
        list(UnstructuredParser().lazy_parse(Blob(data=b"x", mimetype=PPTX_MIME, path="a.pptx")))
    assert list(isolated_tmp.iterdir()) == []


@pytest.mark.parametrize("ext", [".odp", ".ods", ".odg"])
def test_odf_formats_unknown_to_unstructured_are_not_forced(spy_partition, isolated_tmp, ext):
    """unstructured has no partitioner for these: forcing their MIME type would only hide detection."""
    list(UnstructuredParser().lazy_parse(Blob(data=b"PK", mimetype=None, path=f"f{ext}")))
    assert "content_type" not in spy_partition[0]
