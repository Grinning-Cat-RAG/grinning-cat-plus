from __future__ import annotations

import os
import tempfile
from typing import Any, Iterator

from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document


# Formats whose magic bytes resolve to application/zip rather than the real type.
# We force the correct content_type so unstructured picks the right partitioner.
# Only formats unstructured can partition: ODP/ODS/ODG have no partitioner there
# (ODP is converted to PPTX by LibreOfficeConversionParser, ODS goes to OdsParser).
_EXT_TO_MIME: dict[str, str] = {
    ".docx":  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx":  "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx":  "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".odt":   "application/vnd.oasis.opendocument.text",
}
_MIME_TO_EXT: dict[str, str] = {mime: ext for ext, mime in _EXT_TO_MIME.items()}


class UnstructuredParser(BaseBlobParser):
    """General-purpose parser backed by ``unstructured.partition.auto.partition``.

    Passes ``content_type`` explicitly for ZIP-based formats so that
    libmagic mis-detection (returning ``application/zip``) does not prevent
    the correct partitioner from being selected. The blob MIME type wins over the
    source extension: the source may be a URL without extension, or the name of the
    original file when the bytes were converted (e.g. ``deck.odp`` holding PPTX bytes).
    """

    def __init__(self, **partition_kwargs: Any) -> None:
        self._partition_kwargs = partition_kwargs

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        try:
            from unstructured.partition.auto import partition
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "unstructured is required: install the `unstructured[all-docs]` "
                "dependency declared in pyproject.toml."
            ) from exc

        source_suffix = os.path.splitext(blob.source or "")[1].lower()
        forced_mime = blob.mimetype if blob.mimetype in _MIME_TO_EXT else _EXT_TO_MIME.get(source_suffix)
        suffix = _MIME_TO_EXT[forced_mime] if forced_mime else source_suffix
        temp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                temp_path = tmp.name
                tmp.write(blob.as_bytes())

            kwargs: dict[str, Any] = dict(self._partition_kwargs)
            # Force content_type for ZIP-based formats to bypass libmagic mis-detection
            if forced_mime:
                kwargs["content_type"] = forced_mime
            # Always pass the original filename so unstructured adds correct metadata
            kwargs.setdefault("metadata_filename", blob.source or temp_path)

            elements = partition(filename=temp_path, **kwargs)
            is_presentation = forced_mime == _EXT_TO_MIME[".pptx"]
            for element in elements:
                metadata = element.metadata.to_dict() if hasattr(element.metadata, "to_dict") else {}
                # Presentations (PPTX, and PPT / ODP after conversion): unstructured numbers the
                # slides as pages; expose it as ``slide`` too, which the chunkers keep per chunk.
                if is_presentation and "page_number" in metadata:
                    metadata["slide"] = metadata["page_number"]
                yield Document(page_content=str(element), metadata=metadata)
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
