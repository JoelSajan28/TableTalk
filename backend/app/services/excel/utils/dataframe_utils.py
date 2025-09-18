from __future__ import annotations
import pandas as pd
from .names import safe_name

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized, SQLite-safe column names."""
    out = df.copy()
    out.columns = [safe_name(c) for c in out.columns]
    return out
