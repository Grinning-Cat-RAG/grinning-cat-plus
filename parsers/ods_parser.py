import json
from typing import Iterator

import pandas as pd
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document

from .excel_parser import _sanitize


class OdsParser(BaseBlobParser):
    """Parser for ODS (OpenDocument Spreadsheet) files.

    Uses pandas.read_excel with engine="odf" (backed by odfpy, already
    available as a transitive dependency of unstructured[all-docs]).

    Each sheet is yielded as a separate Document whose page_content is
    a JSON-serialised dict (same contract as ExcelParser / TableParser),
    with an extra ``sheet_name`` metadata field.

    Timestamp keys/values are normalised via _sanitize so that json.dumps
    never raises ``TypeError: keys must be str, int, float, bool or None,
    not Timestamp``.
    """

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        with blob.as_bytes_io() as file:
            sheets: dict[str, pd.DataFrame] = pd.read_excel(
                file, sheet_name=None, index_col=0, engine="odf"
            )

        for sheet_name, df in sheets.items():
            content = _sanitize(df.to_dict())
            metadata = dict(blob.metadata or {})
            metadata["sheet_name"] = str(sheet_name)
            yield Document(page_content=json.dumps(content), metadata=metadata)
