from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document


class RawHTMLParser(BaseBlobParser):
    """Parser that preserves raw HTML content instead of stripping tags.

    Used when ``HTMLSemanticChunker`` is the active chunker, so the
    chunker receives the original HTML and can split semantically on
    heading structure (h1, h2, …) rather than on plain text.
    """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        yield Document(
            page_content=blob.as_string(),
            metadata={"source": blob.source},
        )