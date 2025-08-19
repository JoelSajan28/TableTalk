from pathlib import Path
import pandas as pd
import sqlite3
from app.db.sqlite import SQLITE_PATH

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def ingest_excel_to_sqlite(path: Path, dataset: str) -> dict:
    conn = sqlite3.connect(SQLITE_PATH)
    xls = pd.ExcelFile(path)

    tables = []
    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet).dropna(how="all")
        df = _normalize_cols(df)
        table_name = f"{dataset}__{sheet.strip().lower().replace(' ', '_')}"
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        tables.append({"name": table_name, "columns": list(map(str, df.columns)), "rows": int(len(df))})

    # metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tables_metadata (
            dataset TEXT,
            name TEXT,
            columns TEXT,
            row_count INTEGER
        )
    """)
    conn.execute("DELETE FROM tables_metadata WHERE dataset=?", (dataset,))
    for t in tables:
        conn.execute(
            "INSERT INTO tables_metadata (dataset, name, columns, row_count) VALUES (?, ?, ?, ?)",
            (dataset, t["name"], ",".join(t["columns"]), t["rows"])
        )
    conn.commit()
    conn.close()

    return {"dataset": dataset, "tables": tables}
