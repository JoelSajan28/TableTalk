from pathlib import Path
from typing import Dict, Any
import pandas as pd
from sqlalchemy.engine import Engine
from app.db.sqlite import engine
import re

def _safe_name(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "table"
    # Avoid names starting with digits
    if s[0].isdigit():
        s = f"t_{s}"
    return s

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_safe_name(str(c)) for c in df.columns]
    return df

def ingest_excel_to_sqlite(
    file_path: Path,
    dataset: str,
    sql_engine: Engine = engine,
) -> Dict[str, Any]:
    
    xls = pd.ExcelFile(file_path)
    ds = _safe_name(dataset)

    summary = {"dataset": ds, "tables": []}
    # Create a metadata table (if not exists)
    with sql_engine.begin() as conn:
        conn.exec_driver_sql("""
            CREATE TABLE IF NOT EXISTS tables_metadata (
                dataset TEXT,
                worksheet TEXT,
                table_name TEXT,
                row_count INTEGER,
                columns TEXT
            )
        """)
        # Optional: clear old metadata for this dataset
        conn.exec_driver_sql("DELETE FROM tables_metadata WHERE dataset = ?", (ds,))

    for sheet in xls.sheet_names:
        df = pd.read_excel(file_path, sheet_name=sheet, dtype=str)
        df = df.dropna(how="all")
        df = _normalize_columns(df)

        table_name = f"{ds}__{_safe_name(sheet)}"
        df.to_sql(table_name, con=sql_engine, if_exists="replace", index=False)

        with sql_engine.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO tables_metadata (dataset, worksheet, table_name, row_count, columns) VALUES (?, ?, ?, ?, ?)",
                (ds, sheet, table_name, int(len(df)), ",".join(map(str, df.columns))),
            )

        summary["tables"].append({
            "worksheet": sheet,
            "table": table_name,
            "row_count": int(len(df)),
            "columns": list(map(str, df.columns)),
        })

    return summary
