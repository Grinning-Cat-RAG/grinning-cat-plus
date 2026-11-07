"""``slide`` on every chunk of a presentation, whatever the chunker.

Invariant: every chunk built from presentation elements carries a top-level ``slide``, and all
of its content comes from that slide (no chunk spans two slides). Documents without ``slide``
(PDF, DOCX, ...) keep their previous behaviour.
"""
from __future__ import annotations

import asyncio
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from langchain_core.documents import Document

from tests.test_rabbithole import rabbithole  # noqa: F401 - fixture: imports the plugin with the core stubbed


@pytest.fixture(scope="module")
def custom(rabbithole):  # noqa: F811
    import grinning_cat_plus.chunkers.custom as custom

    return custom


class ShufflingSplitter:
    """Semantic-splitter double: clusters its input arbitrarily (like similarity clustering,
    which may group non-adjacent elements) and returns the same shape as the real one."""

    def __init__(self, seed: int, extra_keys: bool = False):
        self.rng = random.Random(seed)
        self.calls: list[list[dict]] = []
        self.extra_keys = extra_keys

    def chunk(self, chunks):
        self.calls.append(chunks)
        items = list(chunks)
        self.rng.shuffle(items)
        out = []
        while items:
            n = self.rng.randint(1, len(items))
            group, items = items[:n], items[n:]
            entry = {"text": " ".join(c["text"] for c in group), "metadata": group}
            if self.extra_keys:
                entry.update(has_formula=False, formula_count=0)
            out.append(entry)
        return out


def _semantic(custom, kind, splitter):
    if kind == "semantic":
        chunker = custom.SemanticChunker(cluster_threshold=0.5, similarity_threshold=0.5, max_tokens=100)
    else:
        chunker = custom.MathAwareSemanticChunker.__new__(custom.MathAwareSemanticChunker)
        custom.BaseChunker.__init__(chunker)
    chunker._splitter = splitter
    return chunker


def _elements(slides: list[int]) -> list[Document]:
    return [
        Document(page_content=f"s{slide}e{i}", metadata={"slide": slide, "page_number": slide, "source": "d.pptx"})
        for i, slide in enumerate(slides)
    ]


def _source_slides(chunk: Document) -> set:
    return {c["metadata"].get("slide") for c in chunk.metadata["source_chunks"]}


@pytest.mark.parametrize("kind", ["semantic", "math"])
@settings(max_examples=60, deadline=None)
@given(
    runs=st.lists(st.tuples(st.integers(1, 9), st.integers(1, 4)), min_size=1, max_size=6),
    seed=st.integers(0, 10_000),
)
def test_semantic_chunks_never_span_slides(custom, kind, runs, seed):
    slides = [slide for slide, n in runs for _ in range(n)]
    chunker = _semantic(custom, kind, ShufflingSplitter(seed, extra_keys=kind == "math"))
    chunks = asyncio.run(chunker.split_documents(_elements(slides)))

    assert chunks
    for chunk in chunks:
        assert _source_slides(chunk) == {chunk.metadata["slide"]}
    # nothing lost, nothing duplicated
    got = sorted(c["text"] for chunk in chunks for c in chunk.metadata["source_chunks"])
    assert got == sorted(f"s{s}e{i}" for i, s in enumerate(slides))


@pytest.mark.parametrize("kind", ["semantic", "math"])
def test_documents_without_slide_are_chunked_together_as_before(custom, kind):
    splitter = ShufflingSplitter(1, extra_keys=kind == "math")
    chunker = _semantic(custom, kind, splitter)
    docs = [Document(page_content=f"p{i}", metadata={"page_number": i}) for i in range(5)]
    chunks = asyncio.run(chunker.split_documents(docs))
    assert len(splitter.calls) == 1 and len(splitter.calls[0]) == 5
    assert all("slide" not in chunk.metadata for chunk in chunks)


def test_math_semantic_keeps_its_formula_fields(custom):
    chunker = _semantic(custom, "math", ShufflingSplitter(3, extra_keys=True))
    [chunk, *_] = asyncio.run(chunker.split_documents(_elements([1])))
    assert {"has_formula", "formula_count", "slide", "source_chunks"} <= set(chunk.metadata)


def test_html_semantic_chunker_header_cannot_overwrite_slide(custom):
    chunker = custom.HTMLSemanticChunker(headers_to_split_on=[("h1", "slide")], elements_to_preserve=[])
    doc = Document(page_content="<h1>Titolo</h1><p>alpha beta</p>", metadata={"slide": 3})
    chunks = asyncio.run(chunker.split_documents([doc]))
    assert chunks and all(c.metadata["slide"] == 3 for c in chunks)


def test_html_semantic_chunker_adds_nothing_to_other_documents(custom):
    """Only `slide` is given back: other formats keep the splitter's metadata alone, as before."""
    chunker = custom.HTMLSemanticChunker(headers_to_split_on=[("h1", "Header 1")], elements_to_preserve=[])
    doc = Document(page_content="<h1>Uno</h1><p>alpha</p>", metadata={"page_number": 4, "filename": "a.pdf"})
    chunks = asyncio.run(chunker.split_documents([doc]))
    assert chunks and all(set(c.metadata) <= {"Header 1"} for c in chunks)


def test_html_semantic_chunker_gives_slide_back(custom):
    chunker = custom.HTMLSemanticChunker(headers_to_split_on=[("h1", "Header 1")], elements_to_preserve=[])
    docs = [
        Document(page_content="<h1>Uno</h1><p>alpha beta gamma</p>", metadata={"slide": 1, "source": "d.pptx"}),
        Document(page_content="<h1>Due</h1><p>delta epsilon</p>", metadata={"slide": 2, "source": "d.pptx"}),
    ]
    chunks = asyncio.run(chunker.split_documents(docs))
    assert chunks
    for chunk in chunks:
        expected = 1 if any(w in chunk.page_content for w in ("Uno", "alpha")) else 2
        assert chunk.metadata["slide"] == expected
    assert any(c.metadata.get("Header 1") == "Uno" for c in chunks)  # the splitter's own metadata survives


@pytest.mark.parametrize("name", ["TokenSpacyChunker", "TokenNLTKChunker", "HierarchicalChunker", "MathAwareHierarchicalChunker"])
def test_per_document_chunkers_keep_the_right_slide(custom, name):
    import inspect

    cls = getattr(custom, name)
    defaults = {
        "chunk_size": 200, "chunk_overlap": 0, "max_length": 100_000, "language": "english",
        "min_chunk_size": 1, "max_chunk_size": 200, "preserve_structure": True, "formula_context_window": 50,
    }
    params = inspect.signature(cls.__init__).parameters
    chunker = cls(**{k: v for k, v in defaults.items() if k in params})
    if name == "TokenNLTKChunker":
        # punkt_tab is downloaded at runtime (see test_rabbithole), not in the test environment:
        # a period-based sentence splitter exercises the same metadata path.
        chunker.splitter._tokenizer = lambda text, language=None: [t for t in text.split(". ") if t]
    docs = [
        Document(page_content=text, metadata={"slide": slide, "source": "d.pptx"})
        for text, slide in (("alpha uno", 1), ("beta uno", 1), ("gamma due", 2))
    ]
    chunks = asyncio.run(chunker.split_documents(docs))
    assert chunks
    for chunk in chunks:
        expected = {1} if "uno" in chunk.page_content else {2}
        assert "uno" not in chunk.page_content or "due" not in chunk.page_content
        assert {chunk.metadata["slide"]} == expected
