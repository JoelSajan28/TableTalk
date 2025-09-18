# app/services/excel_to_sqlite.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, TypedDict

import pandas as pd

from app.db.sqlite import get_engine_for
from app.services.excel_preprocessor import ExcelPreprocessor, Diagnostic
from app.services.common import _safe_name
from app.services.dataframe_service import _normalize_cols
from app.constants.sql_query import (
    SQL_CLEAR_METADATA,
    SQL_CREATE_METADATA,
    SQL_UPSERT_METADATA,
)


class IngestTable(TypedDict):
    name: str
    columns: List[str]
    rows: int


def _unique_name(base: str, used: Set[str]) -> str:
    """Return a unique table name by suffixing _2, _3, ... if needed."""
    name = base or "table"
    idx = 2
    while name in used:
        name = f"{base}_{idx}"
        idx += 1
    used.add(name)
    return name


def ingest_excel_to_sqlite(xlsx_path: Path, dataset: str) -> Dict[str, object]:
    """
    Ingest Excel workbook into SQLite with metadata tracking.
    Returns:
      {
        filename, dataset, sqlite_path,
        tables: [{name, columns, rows}, ...],
        diagnostics: {
            items: [ {dataset,sheet,table_name,severity,code,message,handled,suggestion}, ... ],
            summary: {info: n, warning: n, error: n}
        }
      }
    """
    engine, db_path = get_engine_for(dataset, fresh=True)
    ds_key = _safe_name(dataset)

    prep = ExcelPreprocessor(
        gap_rows_as_split=2,
        drop_all_null_cols=False,
        drop_all_null_rows=False,
        trim_text=False,
        normalize_text_lower=False,
        parse_dates=False,
        infer_numeric=False,
        infer_bool=False,
    )

    tables = prep.process_workbook(xlsx_path, ds_key)

    diags: List[Diagnostic] = prep.diagnostics
    diag_items = [d.__dict__ for d in diags]
    summary = {"info": 0, "warning": 0, "error": 0}
    for d in diags:
        if d.severity in summary:
            summary[d.severity] += 1

    out_tables: List[IngestTable] = []
    with engine.begin() as conn:
        conn.execute(SQL_CREATE_METADATA)
        conn.execute(SQL_CLEAR_METADATA)

        used_names: Set[str] = set()

        for t in tables:
            base = _safe_name(t.name) or "table"
            name = _unique_name(base, used_names)

            df: pd.DataFrame = _normalize_cols(t.df)
            if df is None or df.empty or df.shape[1] == 0:
                continue

            
            df.to_sql(name, conn.connection, if_exists="replace", index=False)

            cols = [str(c) for c in df.columns]
            rows = int(df.shape[0])
            out_tables.append({"name": name, "columns": cols, "rows": rows})

            conn.execute(
                SQL_UPSERT_METADATA,
                {"d": ds_key, "n": name, "c": ",".join(cols), "r": rows},
            )

    return {
        "filename": xlsx_path.name,
        "dataset": ds_key,
        "sqlite_path": str(db_path),
        "tables": out_tables,
        "diagnostics": {
            "items": diag_items,
            "summary": summary,
        },
    }
