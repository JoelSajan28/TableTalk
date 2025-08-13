# app/services/excel_split.py
from pathlib import Path
from typing import Dict, Any
import pandas as pd

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df

def split_excel(path: Path, preview_rows: int = 5) -> Dict[str, Any]:
    xls = pd.ExcelFile(path)
    result: Dict[str, Any] = {"sheets": []}

    for sheet in xls.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet, dtype=str)
        df = df.dropna(how="all")
        df = _normalize_columns(df)
        result["sheets"].append({
            "sheet": sheet,
            "row_count": int(len(df)),
            "columns": list(df.columns),
            "preview": df.head(preview_rows).to_dict(orient="records"),
        })

    result["total_sheets"] = len(result["sheets"])
    return result
