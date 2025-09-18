from __future__ import annotations
import re
from typing import Dict
import pandas as pd

_VAGUE = re.compile(r"(sheet|sheet\d+|sheet_\d+|table|data)\s*\d*$", re.I)
_NUM_LIKE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")

def is_vague_sheet_name(name: str) -> bool:
    return bool(_VAGUE.fullmatch(name.strip()))

def looks_like_vertical_headers(df: pd.DataFrame) -> bool:
    if df.empty or df.shape[1] == 0:
        return False
    first_col = df.iloc[:, 0]
    strings_ratio = first_col.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
    top_row_nulls = df.iloc[0].isna().mean()
    return strings_ratio >= 0.7 and top_row_nulls >= 0.5

def mixed_type_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    TRUE mixed-type columns (or object columns mixing numeric-looking and non-numeric strings).
    If all values are numeric-looking strings, do NOT warn.
    """
    out: Dict[str, str] = {}
    for c in df.columns:
        series = df[c].dropna()
        if series.empty:
            continue
        sample = series.head(500)
        types = set(type(x).__name__ for x in sample)
        if len(types) > 1:
            out[c] = ", ".join(sorted(types))
        else:
            if sample.dtype == object:
                s = sample.astype(str)
                is_num_like = s.str.match(_NUM_LIKE, na=False)
                frac_num_like = is_num_like.mean()
                if 0 < frac_num_like < 1:
                    out[c] = "text with mixed numeric-looking and non-numeric values"
    return out