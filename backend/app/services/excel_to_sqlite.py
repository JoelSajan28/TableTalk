# app/services/excel_to_sqlite.py
from pathlib import Path
import re
import pandas as pd
from sqlalchemy import text
from app.db.sqlite import get_engine_for

def _safe_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "_", s)
    return s

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_safe_name(c) for c in df.columns]
    return df

def ingest_excel_to_sqlite(xlsx_path: Path, dataset: str) -> dict:
    engine, db_path = get_engine_for(dataset)
    ds_key = _safe_name(dataset)
    xls = pd.ExcelFile(xlsx_path)

    tables = []
    with engine.begin() as conn:
        # Ensure metadata table exists and can't duplicate per name
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS tables_metadata (
                dataset   TEXT NOT NULL,
                name      TEXT NOT NULL,
                columns   TEXT,
                row_count INTEGER,
                UNIQUE(name)
            )
        """))

        # Because each dataset has its own DB, wipe all metadata in this DB
        conn.execute(text("DELETE FROM tables_metadata"))

        for sheet in xls.sheet_names:
            df = pd.read_excel(xlsx_path, sheet_name=sheet)  # keep numeric types
            df = _normalize_cols(df)

            table_name = _safe_name(sheet)  # (no dataset prefix since each dataset is its own DB)
            df.to_sql(table_name, conn.connection, if_exists="replace", index=False)

            cols = list(map(str, df.columns))
            rows = int(len(df))
            tables.append({"name": table_name, "columns": cols, "rows": rows})

            # Upsert into metadata for this DB
            conn.execute(
                text("""
                    INSERT OR REPLACE INTO tables_metadata (dataset, name, columns, row_count)
                    VALUES (:d, :n, :c, :r)
                """),
                {"d": ds_key, "n": table_name, "c": ",".join(cols), "r": rows},
            )

    return {"dataset": ds_key, "sqlite_path": str(db_path), "tables": tables}
