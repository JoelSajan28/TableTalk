from __future__ import annotations
from typing import Optional, List
from rapidfuzz import process, fuzz
import pandas as pd

def pick_island_column(df: pd.DataFrame, candidates: Optional[List[str]] = None) -> Optional[str]:
    """
    Try to find an 'Island' / region column even if named weirdly.
    Returns the column name or None.
    """
    if candidates is None:
        candidates = [
            "island", "islands", "region", "county", "area", "province",
            "location", "territory", "state"
        ]
    # direct match first
    for col in df.columns:
        if str(col).strip().lower() in candidates:
            return col

    # fuzzy match against "island"
    choices = list(map(str, df.columns))
    if not choices:
        return None
    match, score, _ = process.extractOne("island", choices, scorer=fuzz.WRatio)
    return match if score >= 80 else None
