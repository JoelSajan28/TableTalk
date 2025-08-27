# app/services/excel_to_sqlite.py
from pathlib import Path
import re
import pandas as pd
from sqlalchemy import text
from app.db.sqlite import get_engine_for
from app.services.excel_preprocessor import ExcelPreprocessor 

def _safe_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    s = s.strip("_") or "table"
    return s

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_safe_name(c) for c in df.columns]
    return df

def ingest_excel_to_sqlite(xlsx_path: Path, dataset: str) -> dict:
    engine, db_path = get_engine_for(dataset)
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

    out_tables = []
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tables_metadata (
                dataset   TEXT NOT NULL,
                name      TEXT NOT NULL,
                columns   TEXT,
                row_count INTEGER,
                UNIQUE(name)
            )
        """))
        conn.execute(text("DELETE FROM tables_metadata"))

        used_names = set()
        for t in tables:
            base = _safe_name(t.name)
            name = base
            i = 2
            while name in used_names:
                name = f"{base}_{i}"
                i += 1
            used_names.add(name)

            df = _normalize_cols(t.df)
            if df.shape[1] == 0 or df.shape[0] == 0:
                continue

            df.to_sql(name, conn.connection, if_exists="replace", index=False)

            cols = list(map(str, df.columns))
            rows = int(len(df))
            out_tables.append({"name": name, "columns": cols, "rows": rows})

            conn.execute(
                text("""INSERT OR REPLACE INTO tables_metadata (dataset, name, columns, row_count)
                        VALUES (:d,:n,:c,:r)"""),
                {"d": ds_key, "n": name, "c": ",".join(cols), "r": rows},
            )

    return {
        "filename": xlsx_path.name,
        "dataset": ds_key,
        "sqlite_path": str(db_path),
        "tables": out_tables,
    }
