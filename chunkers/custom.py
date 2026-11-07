import asyncio
import itertools
import json
from typing import Any, Dict, List, Iterable, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import (
    HTMLSemanticPreservingSplitter,
    RecursiveJsonSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter,
)
from cat import BaseChunker, EmbeddedBaseChunker

from .semantic_chunker import SemanticChunker as SemanticSplitter, MathAwareSemanticChunker as MathAwareSemanticSplitter
from .hierarchical_chunker import (
    HierarchicalChunker as HierarchicalSplitter,
    MathAwareHierarchicalChunker as MathAwareHierarchicalSplitter,
)


def _slide_runs(documents: Iterable[Document]) -> List[Tuple[Any, List[Dict[str, Any]]]]:
    """Group consecutive documents by their ``slide`` (None when absent), as splitter input.

    Semantic chunkers cluster by similarity across their whole input, so a chunk could mix
    non-adjacent slides: chunking each slide on its own keeps every chunk within one slide.
    Documents without ``slide`` form runs of None, i.e. one input as before.
    """
    return [
        (slide, [{"text": doc.page_content, "metadata": doc.metadata} for doc in run])
        for slide, run in itertools.groupby(documents, key=lambda doc: doc.metadata.get("slide"))
    ]


def _with_slide(metadata: Dict[str, Any], slide: Any) -> Dict[str, Any]:
    return metadata if slide is None else {**metadata, "slide": slide}


class SemanticChunker(EmbeddedBaseChunker):
    def __init__(self, cluster_threshold: float, similarity_threshold: float, max_tokens: int):
        super().__init__()

        self._cluster_threshold = cluster_threshold
        self._similarity_threshold = similarity_threshold
        self._max_tokens = max_tokens
        self._analyzer = None

    def _get_splitter(self) -> SemanticSplitter:
        return SemanticSplitter(
            cluster_threshold=self._cluster_threshold,
            similarity_threshold=self._similarity_threshold,
            max_tokens=self._max_tokens
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        await self._ensure_embedder()

        return [
            Document(
                page_content=chunk["text"],
                metadata=_with_slide({"source_chunks": chunk.get("metadata", [])}, slide),
            )
            for slide, texts in _slide_runs(documents)
            for chunk in self.splitter.chunk(texts)
        ]


class HTMLSemanticChunker(BaseChunker):
    def __init__(self, headers_to_split_on: List[Tuple[str, str]] | List[List[str]], elements_to_preserve: List[str]):
        super().__init__()

        self._headers_to_split_on = headers_to_split_on if isinstance(headers_to_split_on[0], tuple) else [
            (header, header) for header in headers_to_split_on
        ]
        self._elements_to_preserve = elements_to_preserve

    def _get_splitter(self) -> HTMLSemanticPreservingSplitter:
        return HTMLSemanticPreservingSplitter(
            headers_to_split_on=self._headers_to_split_on,
            separators=["\n\n", "\n", ". ", "! ", "? "],
            max_chunk_size=50,
            preserve_images=True,
            preserve_videos=True,
            elements_to_preserve=self._elements_to_preserve,
            denylist_tags=["script", "style", "head"],
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        # split_text works on the text only: give `slide` back (presentations), applied last so a
        # header named "slide" cannot overwrite it. Other metadata stays the splitter's own.
        return await asyncio.to_thread(
            lambda: [
                Document(page_content=chunk.page_content, metadata=_with_slide(chunk.metadata, doc.metadata.get("slide")))
                for doc in docs
                for chunk in self.splitter.split_text(doc.page_content)
            ]
        )


class JSONChunker(BaseChunker):
    def __init__(self, max_chunk_size: int, min_chunk_size: int | None = None):
        super().__init__()

        self._max_chunk_size = max_chunk_size
        self._min_chunk_size = min_chunk_size

    def _get_splitter(self) -> RecursiveJsonSplitter:
        return RecursiveJsonSplitter(max_chunk_size=self._max_chunk_size, min_chunk_size=self._min_chunk_size)

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        texts = [json.loads(doc.page_content) for doc in docs]
        metadata = [doc.metadata for doc in docs]
        return await asyncio.to_thread(self.splitter.create_documents, texts, metadata)


class TokenSpacyChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int, max_length: int):
        super().__init__()

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._max_length = max_length

    def _get_splitter(self) -> SpacyTextSplitter:
        return SpacyTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap, max_length=self._max_length
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(self.splitter.split_documents, docs)


class TokenNLTKChunker(BaseChunker):
    def __init__(self, chunk_size: int, chunk_overlap: int, language: str):
        super().__init__()

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._language = language

    def _get_splitter(self) -> NLTKTextSplitter:
        return NLTKTextSplitter(
            chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap, language=self._language
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(self.splitter.split_documents, docs)


class HierarchicalChunker(BaseChunker):
    def __init__(
        self, chunk_size: int, chunk_overlap: int, min_chunk_size: int, max_chunk_size: int, preserve_structure: bool,
    ):
        super().__init__()

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._preserve_structure = preserve_structure

    def _get_splitter(self) -> HierarchicalSplitter:
        return HierarchicalSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_chunk_size=self._min_chunk_size,
            max_chunk_size=self._max_chunk_size,
            preserve_structure=self._preserve_structure
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(
            lambda: [chunk for doc in docs for chunk in self.splitter.chunk_document(doc.page_content, doc.metadata)]
        )


class MathAwareHierarchicalChunker(BaseChunker):
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_size: int,
        max_chunk_size: int,
        formula_context_window: int,
        preserve_structure: bool,
    ):
        super().__init__()

        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._max_chunk_size = max_chunk_size
        self._formula_context_window = formula_context_window
        self._preserve_structure = preserve_structure

    def _get_splitter(self) -> MathAwareHierarchicalSplitter:
        return MathAwareHierarchicalSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            min_chunk_size=self._min_chunk_size,
            max_chunk_size=self._max_chunk_size,
            formula_context_window=self._formula_context_window,
            preserve_structure=self._preserve_structure,
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        docs = list(documents)
        return await asyncio.to_thread(
            lambda: [chunk for doc in docs for chunk in self.splitter.chunk_document(doc.page_content, doc.metadata)]
        )


class MathAwareSemanticChunker(EmbeddedBaseChunker):
    def __init__(
        self,
        cluster_threshold: float,
        similarity_threshold: float,
        max_tokens: int,
        formula_context_boost: float = 0.3,
    ):
        super().__init__()

        self._cluster_threshold = cluster_threshold
        self._similarity_threshold = similarity_threshold
        self._max_tokens = max_tokens
        self._formula_context_boost = formula_context_boost

    def _get_splitter(self) -> MathAwareSemanticSplitter:
        return MathAwareSemanticSplitter(
            cluster_threshold=self._cluster_threshold,
            similarity_threshold=self._similarity_threshold,
            max_tokens=self._max_tokens,
            formula_context_boost=self._formula_context_boost,
        )

    async def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        await self._ensure_embedder()

        return [
            Document(
                page_content=chunk["text"],
                metadata=_with_slide(
                    {
                        "source_chunks": chunk.get("metadata", []),
                        "has_formula": chunk.get("has_formula", False),
                        "formula_count": chunk.get("formula_count", 0),
                    },
                    slide,
                ),
            )
            for slide, texts in _slide_runs(documents)
            for chunk in self.splitter.chunk(texts)
        ]
