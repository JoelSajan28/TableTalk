from __future__ import annotations
from typing import List, Tuple
import pandas as pd

def split_sheet_into_tables(
    raw: pd.DataFrame,
    gap_rows_as_split: int = 2,
    single_value_row_is_separator: bool = True,
) -> List[pd.DataFrame]:
    """
    Detect multiple tables separated by gaps.
    Gap rows are:
      - entirely blank rows, OR
      - (if enabled) rows with <= 1 non-null cell (section/separator rows).
    Detect BEFORE any filling so separators remain visible.
    """
    raw = raw.copy()

    is_blank = raw.isna().all(axis=1)
    non_null_counts = raw.notna().sum(axis=1)
    is_single_value = non_null_counts <= 1 if single_value_row_is_separator else pd.Series(False, index=raw.index)

    gap_mask = (is_blank | is_single_value)

    blocks: List[Tuple[int, int]] = []
    start = 0
    i = 0
    while i < len(raw):
        if gap_mask.iloc[i:i + gap_rows_as_split].all().all():
            end = i
            if end > start:
                blocks.append((start, end))
            i += gap_rows_as_split
            start = i
        else:
            i += 1

    if start < len(raw):
        blocks.append((start, len(raw)))
    if not blocks:
        blocks = [(0, len(raw))]

    parts: List[pd.DataFrame] = []
    for a, b in blocks:
        chunk = raw.iloc[a:b].reset_index(drop=True)
        chunk = chunk.dropna(how="all")
        if not chunk.empty:
            parts.append(chunk.reset_index(drop=True))

    return parts
