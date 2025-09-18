from __future__ import annotations
from typing import Any, Dict, List

# ---- Heuristics ----

_METRIC_HINTS = (
    "amount", "total", "sum", "count", "score", "value",
    "revenue", "price", "qty", "quantity"
)

def is_metric_like(col: str) -> bool:
    cl = (col or "").lower()
    return any(k in cl for k in _METRIC_HINTS)

# ---- Rendering ----

def markdown_table(columns: List[str], rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
    """Render a small Markdown table for the provided rows (at most max_rows)."""
    if not rows:
        return ""
    head = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [head, sep]
    for r in rows[:max_rows]:
        vals = [str(r.get(c, "")) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"\n_…{len(rows) - max_rows} more row(s) omitted…_")
    return "\n".join(lines)

def fallback_summary(question: str, columns: List[str], rows: List[Dict[str, Any]], max_items: int = 6) -> str:
    """Deterministic, safe summary if the LLM errors or returns junk."""
    if not rows:
        return (
            "**Answer**\n\n"
            "No matching rows were returned for this query.\n\n"
            "**Next steps**\n"
            "- Check filters or spelling\n- Expand the date range\n- Try a different table or column"
        )

    # choose entity-like and metric-like columns
    entity_col = None
    metric_col = None
    for c in columns:
        lc = c.lower()
        if entity_col is None and any(k in lc for k in (
            "name", "title", "task", "category", "user", "customer", "product", "student", "account", "id"
        )):
            entity_col = c
        if metric_col is None and is_metric_like(c):
            metric_col = c

    bullets: List[str] = []
    sample = rows[:max_items]
    if entity_col and metric_col:
        try:
            sample = sorted(rows, key=lambda r: float(r.get(metric_col) or 0), reverse=True)[:max_items]
        except Exception:
            sample = rows[:max_items]
        for r in sample:
            bullets.append(f"- {r.get(entity_col)} — **{metric_col}**: {r.get(metric_col)}")
    else:
        cols_show = columns[: min(4, len(columns))]
        for r in sample:
            bullets.append("- " + ", ".join(f"{c}={r.get(c)}" for c in cols_show))

    out: List[str] = [
        "**Answer**\n\nHere’s a quick summary based on the returned rows.",
        "\n**Key facts**",
        *bullets,
    ]
    if len(rows) <= 10:
        out.append("\n**Table**\n" + markdown_table(columns, rows))
    return "\n".join(out)

# ---- Prompt building ----

def build_user_prompt(question: str, columns: List[str], rows: List[Dict[str, Any]], preview: int = 50) -> str:
    """Create a compact user content payload for the NL answering model."""
    row_count = len(rows)
    preview_rows = rows[: min(preview, row_count)]
    return (
        f"User question:\n{question}\n\n"
        f"Columns: {columns}\n"
        f"Row count: {row_count}\n"
        f"Rows (preview up to {min(preview, row_count)}):\n{preview_rows}\n\n"
        "Follow SYSTEM instructions exactly."
    )
