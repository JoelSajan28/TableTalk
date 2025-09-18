from __future__ import annotations
from typing import Any, List
import pandas as pd
from .names import normalize_col_name, dedupe_columns

def guess_header_row(df: pd.DataFrame) -> int:
    """Pick the most header-like row among the first 10."""
    best_row = 0
    best_score = -1.0
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        non_null = row.notna().sum()
        textish = sum(1 for v in row if pd.notna(v) and isinstance(v, str) and str(v).strip())
        score = non_null + 0.5 * textish
        if score > best_score:
            best_score = score
            best_row = i
    return best_row

def normalize_text_cells(df: pd.DataFrame, *, trim_text: bool, normalize_text_lower: bool) -> pd.DataFrame:
    def _clean(x: Any) -> Any:
        if isinstance(x, str):
            t = x.strip() if trim_text else x
            return t.lower() if (normalize_text_lower and isinstance(t, str)) else t
        return x
    return df.applymap(_clean)

def fix_header_and_clean(
    chunk: pd.DataFrame,
    *,
    drop_all_null_cols: bool,
    drop_all_null_rows: bool,
    trim_text: bool,
    normalize_text_lower: bool,
) -> pd.DataFrame:
    """Choose a header row, rename columns, drop empty rows/cols, normalize text."""
    df = chunk.copy()
    header_row = guess_header_row(df)
    header = [normalize_col_name(x) for x in df.iloc[header_row].tolist()]
    header = dedupe_columns(header)
    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = header

    if drop_all_null_cols:
        df = df.dropna(axis=1, how="all")
    if drop_all_null_rows:
        df = df.dropna(axis=0, how="all").reset_index(drop=True)

    if trim_text or normalize_text_lower:
        df = normalize_text_cells(df, trim_text=trim_text, normalize_text_lower=normalize_text_lower)
    return df
