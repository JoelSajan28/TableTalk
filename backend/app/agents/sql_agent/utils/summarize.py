from __future__ import annotations
from typing import List, Optional
import re
import pandas as pd

def _pick_entity_column(columns: List[str]) -> Optional[str]:
    priors = ["name", "user", "customer", "client", "account", "student", "id", "title", "product", "task", "category"]
    for p in priors:
        for c in columns:
            if p in c.lower():
                return c
    return columns[0] if columns else None

def _pick_metric_column(df: pd.DataFrame) -> Optional[str]:
    if df.empty:
        return None
    numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        return None
    preferred = ["transaction", "transactions", "amount", "total", "sum", "count", "score", "value"]
    ranked = sorted(
        numeric,
        key=lambda c: (
            -max((1 if key in c.lower() else 0) for key in preferred),
            -df[c].fillna(0).abs().mean()
        )
    )
    return ranked[0] if ranked else None

def summarize_rows(question: str, df: pd.DataFrame, max_items: int = 10) -> str:
    if df is None or df.shape[0] == 0:
        return "I didn’t find any matching rows for that request."

    entity_col = _pick_entity_column(list(df.columns))
    metric_col = _pick_metric_column(df)

    view = df.copy()
    if entity_col:
        view[entity_col] = view[entity_col].astype(str)
    top = view.head(max_items)

    if entity_col and metric_col:
        if re.search(r"\b(highest|top|max|most|largest|biggest)\b", (question or "").lower()):
            top = view.sort_values(metric_col, ascending=False).head(max_items)
        bullets = [f"- {r.get(entity_col)} — {metric_col}: {r.get(metric_col)}" for _, r in top.iterrows()]
        count = int(df.shape[0])
        lead = f"Found {count} row(s). Here are the top {len(bullets)}:"
        return "\n".join([lead, *bullets])

    cols_show = list(top.columns)[:4]
    lines = ["Found {} row(s). Sample:".format(int(df.shape[0]))]
    for _, r in top.iterrows():
        pieces = [f"{c}={r.get(c)}" for c in cols_show]
        lines.append("- " + ", ".join(pieces))
    return "\n".join(lines)
