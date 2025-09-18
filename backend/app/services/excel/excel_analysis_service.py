from __future__ import annotations
from typing import Any, Dict, Optional
from collections import Counter

import pandas as pd

from app.services.excel.utils.styles import read_excel_with_styles
from app.services.excel.utils.fuzzy import pick_island_column

def analyze_excel(path: str, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Read an Excel sheet with style-aware header inference, then return
    a compact analysis payload including color usage and a grouped summary
    by a location-like column (if found).
    """
    df, color_map = read_excel_with_styles(path, sheet_name=sheet_name)

    # Find an island-ish column and run a quick groupby
    island_col = pick_island_column(df)
    by_island = None
    if island_col:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            by_island = df.groupby(island_col)[numeric_cols].agg(["count", "mean", "sum"])
        else:
            by_island = df.groupby(island_col).size().to_frame("rows")

    # Count background colors used in the data region
    color_counts = Counter(color_map.values())
    most_common_colors = color_counts.most_common(10)

    return {
        "columns": list(df.columns),
        "preview": df.head(10).to_dict(orient="records"),
        "island_column": island_col,
        "by_island_summary": None if by_island is None else by_island.head(20).to_dict(),
        "color_usage_top10": most_common_colors,
        "notes": [
            "Header row inferred with heuristics (non-empty density, text-likeness, bold, fill).",
            "Colors are ARGB hex from Excel fills; theme/tint nuances may need extra handling.",
            "Rename/clean headers with libraries like `pyjanitor` or `python-slugify` if needed."
        ]
    }
