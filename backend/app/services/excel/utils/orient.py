from __future__ import annotations
import pandas as pd

def maybe_transpose(chunk: pd.DataFrame) -> pd.DataFrame:
    """If the first column looks header-like more than any row, transpose."""
    if chunk.empty:
        return chunk

    def _row_header_score(df: pd.DataFrame, r: int) -> float:
        row = df.iloc[r]
        non_null = row.notna().mean()
        textish = row.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
        uniq = len(pd.Series([str(v) for v in row if pd.notna(v)]).unique()) / (len(row) or 1)
        return 0.5 * non_null + 0.35 * textish + 0.15 * uniq

    def _col_header_score(df: pd.DataFrame, c: int) -> float:
        col = df.iloc[:, c]
        non_null = col.notna().mean()
        textish = col.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
        uniq = len(pd.Series([str(v) for v in col if pd.notna(v)]).unique()) / (len(col) or 1)
        return 0.5 * non_null + 0.35 * textish + 0.15 * uniq

    max_check = min(5, len(chunk))
    row_scores = [_row_header_score(chunk, r) for r in range(max_check)]
    best_row_score = max(row_scores) if row_scores else 0.0
    first_col_score = _col_header_score(chunk, 0) if chunk.shape[1] > 0 else 0.0

    if first_col_score >= best_row_score * 1.15 and first_col_score >= 0.4:
        return chunk.T.reset_index(drop=True)
    return chunk
