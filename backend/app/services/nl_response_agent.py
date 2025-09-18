# app/services/nl_response_agent.py

from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.services.ollama_client import _ollama_chat
from app.constants.system_nl_answer import SYSTEM_NL_ANSWER
from app.constants.system_nl_chatty import SYSTEM_NL_ANSWER_CHATTY

# ------- Utilities -------

_METRIC_HINTS = ("amount", "total", "sum", "count", "score", "value", "revenue", "price", "qty", "quantity")

def _is_metric_like(col: str) -> bool:
    cl = col.lower()
    return any(k in cl for k in _METRIC_HINTS)

def _markdown_table(columns: List[str], rows: List[Dict[str, Any]], max_rows: int = 10) -> str:
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

def _fallback_summary(question: str, columns: List[str], rows: List[Dict[str, Any]], max_items: int = 6) -> str:
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
        if entity_col is None and any(k in lc for k in ("name", "title", "task", "category", "user", "customer", "product", "student", "account", "id")):
            entity_col = c
        if metric_col is None and _is_metric_like(c):
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

    out = [
        "**Answer**\n\nHere’s a quick summary based on the returned rows.",
        "\n**Key facts**",
        *bullets,
    ]
    if len(rows) <= 10:
        out.append("\n**Table**\n" + _markdown_table(columns, rows))
    return "\n".join(out)

def _build_user_prompt(question: str, columns: List[str], rows: List[Dict[str, Any]], preview: int = 50) -> str:
    row_count = len(rows)
    preview_rows = rows[: min(preview, row_count)]
    return (
        f"User question:\n{question}\n\n"
        f"Columns: {columns}\n"
        f"Row count: {row_count}\n"
        f"Rows (preview up to {min(preview, row_count)}):\n{preview_rows}\n\n"
        "Follow SYSTEM instructions exactly."
    )

# ------- Public API -------

def nlp_answer_from_sql(
    *,
    question: str,
    columns: List[str],
    rows: List[Dict[str, Any]],
    model: Optional[str] = None,
    allow_table: bool = True,
    tone: str = "chatty",  # "chatty" | "precise"
) -> str:
    """
    Turn SQL rows into a conversational, grounded answer.
    tone="chatty" -> ChatGPT-like (sections, next steps)
    tone="precise" -> terse analyst style (original prompt)
    """
    if rows is None or columns is None:
        return "**Answer**\n\nNo result set was provided to generate an answer."

    if len(rows) == 0:
        return (
            "**Answer**\n\nNo matching rows were returned for this query.\n\n"
            "**Next steps**\n- Check filters or spelling\n- Expand the date range\n- Try a different table or column"
        )

    system_prompt = SYSTEM_NL_ANSWER_CHATTY if tone == "chatty" else SYSTEM_NL_ANSWER

    try:
        user_prompt = _build_user_prompt(question, columns, rows)
        raw = _ollama_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2 if tone == "chatty" else 0.1,
            model=model,
        ) or ""

        answer = (raw or "").strip()

        # auto-append a small table if not present and rows are small
        if allow_table and len(rows) <= 10 and ("|" not in answer):
            answer = f"{answer}\n\n**Table**\n{_markdown_table(columns, rows, max_rows=10)}"

        if len(answer) > 25_000:
            answer = answer[:25_000] + "\n\n…truncated…"

        # very defensive: if model returned something empty/odd, fallback
        return answer or _fallback_summary(question, columns, rows)

    except Exception:
        return _fallback_summary(question, columns, rows)
