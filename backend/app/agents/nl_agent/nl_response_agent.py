from __future__ import annotations
from typing import Any, Dict, List, Optional

from app.agents.ollama_client import _ollama_chat
from app.constants.system_nl_answer import SYSTEM_NL_ANSWER
from app.constants.system_nl_chatty import SYSTEM_NL_ANSWER_CHATTY

# utils
from app.agents.nl_agent.utils.nl_answer_utils import (
    build_user_prompt,
    markdown_table,
    fallback_summary,
)

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
        user_prompt = build_user_prompt(question, columns, rows)
        raw = _ollama_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2 if tone == "chatty" else 0.1,
            model=model,
        ) or ""

        answer = (raw or "").strip()

        # Auto-append a small table if not present and the dataset is small
        if allow_table and len(rows) <= 10 and ("|" not in answer):
            answer = f"{answer}\n\n**Table**\n{markdown_table(columns, rows, max_rows=10)}"

        if len(answer) > 25_000:
            answer = answer[:25_000] + "\n\n…truncated…"

        return answer or fallback_summary(question, columns, rows)

    except Exception:
        return fallback_summary(question, columns, rows)
