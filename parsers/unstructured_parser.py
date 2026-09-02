from __future__ import annotations

import os
import tempfile
from typing import Any, Iterator

from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document


# Formats whose magic bytes resolve to application/zip rather than the real type.
# We force the correct content_type so unstructured picks the right partitioner.
_EXT_TO_MIME: dict[str, str] = {
    ".docx":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx":  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odp":   "application/vnd.oasis.opendocument.presentation",
    ".odt":   "application/vnd.oasis.opendocument.text",
    ".ods":   "application/vnd.oasis.opendocument.spreadsheet",
    ".odg":   "application/vnd.oasis.opendocument.graphics",
}


class UnstructuredParser(BaseBlobParser):
    """General-purpose parser backed by ``unstructured.partition.auto.partition``.

    Passes ``content_type`` explicitly for ZIP-based formats so that
    libmagic mis-detection (returning ``application/zip``) does not prevent
    the correct partitioner from being selected.
    """

    def __init__(self, **partition_kwargs: Any) -> None:
        self._partition_kwargs = partition_kwargs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "unstructured is required. Add `unstructured[all-docs]>=0.22` "
                "to requirements.txt."
            ) from exc

        suffix = os.path.splitext(blob.source or "")[1].lower()
        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                tmp.write(blob.as_bytes())

            kwargs: dict[str, Any] = dict(self._partition_kwargs)
            # Force content_type for ZIP-based formats to bypass libmagic mis-detection
            forced_mime = _EXT_TO_MIME.get(suffix)
            if forced_mime:
                kwargs["content_type"] = forced_mime
            # Always pass the original filename so unstructured adds correct metadata
            kwargs.setdefault("metadata_filename", blob.source or temp_path)

            elements = partition(filename=temp_path, **kwargs)
            for element in elements:
                yield Document(
                    page_content=str(element),
                    metadata=element.metadata.to_dict() if hasattr(element.metadata, "to_dict")
                    else {},
                )
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
