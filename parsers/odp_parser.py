from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_core.documents import Document

from . import libreoffice

PPTX_MIME = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


class OdpParser(BaseBlobParser):
    """Parse ODP presentations by converting them to PPTX and delegating to ``inner``.

    ``unstructured`` has no ODP partitioner, so the presentation is converted with LibreOffice
    and the PPTX goes through the same parser (and options) used for PowerPoint files.
    The converted blob keeps the original source and metadata. The conversion directory is
    removed before the first document is yielded, so a consumer that stops early leaks nothing.
    """

    def __init__(self, inner: BaseBlobParser, timeout: float = libreoffice.DEFAULT_TIMEOUT_S) -> None:
        self._inner = inner
        self._timeout = timeout

    def _to_pptx(self, blob: Blob) -> Blob:
        with tempfile.TemporaryDirectory(prefix="odp-convert-") as workdir:
            # Fixed ASCII name: the user-supplied source (URL, spaces, non-ASCII) never reaches soffice.
            source = Path(workdir) / "input.odp"
            source.write_bytes(blob.as_bytes())
            pptx = libreoffice.convert(source, "pptx", Path(workdir), timeout=self._timeout)
            data = pptx.read_bytes()
        return Blob(data=data, mimetype=PPTX_MIME, path=blob.source, metadata=dict(blob.metadata or {}))

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        yield from self._inner.lazy_parse(self._to_pptx(blob))
