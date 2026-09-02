import json
from typing import Iterator, Any

import pandas as pd
from langchain_core.document_loaders import BaseBlobParser
from langchain_core.documents.base import Blob, Document


def _sanitize(obj: Any) -> Any:
    """Recursively convert non-JSON-serializable types to strings.

    pandas.DataFrame.to_dict() may produce keys or values of type
    pd.Timestamp, numpy.int64, numpy.float64, etc. that are not
    accepted by json.dumps.  This helper normalises them.
    """
    if isinstance(obj, dict):
        return {_sanitize(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, float) and (obj != obj):   # NaN
        return None
    if hasattr(obj, "isoformat"):                 # datetime, date, Timestamp
        return obj.isoformat()
    if hasattr(obj, "item"):                      # numpy scalars
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


class ExcelParser(BaseBlobParser):
    """Parser for XLSX and XLS files.

    Replaces the core TableParser for Excel MIME types to fix the
    ``TypeError: keys must be str, int, float, bool or None, not Timestamp``
    crash that occurs when a spreadsheet contains date/datetime columns
    or index values (pandas represents them as pd.Timestamp).

    Each sheet is yielded as a separate Document whose page_content is
    a JSON-serialised dict (same contract as TableParser), with an extra
    ``sheet_name`` metadata field.
    """

    _ENGINES: dict[str, str | None] = {
        ".xlsx": None,   # openpyxl (default)
        ".xls":  "xlrd",
    }

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        import os
        suffix = os.path.splitext(blob.source or "")[1].lower()
        engine = self._ENGINES.get(suffix)          # None → pandas default

        with blob.as_bytes_io() as file:
            kwargs: dict[str, Any] = {"sheet_name": None, "index_col": 0}
            if engine:
                kwargs["engine"] = engine
            sheets: dict[str, pd.DataFrame] = pd.read_excel(file, **kwargs)

        for sheet_name, df in sheets.items():
            content = _sanitize(df.to_dict())
            metadata = dict(blob.metadata or {})
            metadata["sheet_name"] = str(sheet_name)
            yield Document(page_content=json.dumps(content), metadata=metadata)
