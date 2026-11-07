from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document

from . import libreoffice

PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_TARGET_MIMES: dict[str, str] = {"pptx": PPTX_MIME, "docx": DOCX_MIME}
_ZIP_MAGIC = b"PK\x03\x04"
# Legacy binary (OLE) formats: if their bytes are a ZIP, the file is already the OOXML target
# (servers often label .docx / .pptx as msword / ms-powerpoint). ODP is a ZIP by nature.
_LEGACY_BINARY_EXTS = frozenset({".doc", ".ppt"})


class LibreOfficeConversionParser(BaseBlobParser):
    """Convert a document with LibreOffice, then delegate the result to ``inner``.

    Used for the formats unstructured cannot read directly: ODP (no partitioner at all) and
    legacy DOC / PPT, which unstructured would convert by launching ``soffice`` itself, with the
    shared profile and no timeout. Here every conversion is bounded and isolated (see
    ``libreoffice.convert``), then the converted file goes through the same parser (and options)
    used for its target format. The converted blob keeps the original source and metadata, and
    the documents keep the original ``filetype`` (the uploaded format, not the intermediate one).
    The conversion directory is removed before the first document is yielded, so a consumer
    that stops early leaks nothing.
    """

    def __init__(
        self,
        inner: BaseBlobParser,
        source_ext: str,
        target_ext: str,
        timeout: float = libreoffice.DEFAULT_TIMEOUT_S,
    ) -> None:
        if target_ext not in _TARGET_MIMES:
            raise ValueError(f"Unsupported conversion target {target_ext!r}: expected one of {sorted(_TARGET_MIMES)}")
        self._inner = inner
        self._source_ext = source_ext
        self._target_ext = target_ext
        self._timeout = timeout

    def _target_blob(self, blob: Blob, data: bytes) -> Blob:
        return Blob(
            data=data,
            mimetype=_TARGET_MIMES[self._target_ext],
            path=blob.source,
            metadata=dict(blob.metadata or {}),
        )

    def _convert(self, blob: Blob) -> Blob:
        with tempfile.TemporaryDirectory(prefix="office-convert-") as workdir:
            # Fixed ASCII name: the user-supplied source (URL, spaces, non-ASCII) never reaches soffice.
            source = Path(workdir) / f"input{self._source_ext}"
            source.write_bytes(blob.as_bytes())
            converted = libreoffice.convert(
                source,
                self._target_ext,
                Path(workdir),
                timeout=self._timeout,
                display_name=blob.source or f"{self._source_ext.lstrip('.').upper()} file",
            )
            data = converted.read_bytes()
        return self._target_blob(blob, data)

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        data = blob.as_bytes()
        if self._source_ext in _LEGACY_BINARY_EXTS and data.startswith(_ZIP_MAGIC):
            yield from self._inner.lazy_parse(self._target_blob(blob, data))
            return
        for doc in self._inner.lazy_parse(self._convert(blob)):
            if "filetype" in doc.metadata:
                doc.metadata["filetype"] = blob.mimetype
            yield doc
