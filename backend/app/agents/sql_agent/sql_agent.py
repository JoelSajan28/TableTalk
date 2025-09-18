from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from app.db.sqlite import get_engine_for
from app.agents.ollama_client import _ollama_chat
from app.agents.nl_agent.nl_response_agent import nlp_answer_from_sql
from app.constants.regex_constants import _SMALLTALK
from app.constants.few_shots import FEW_SHOTS
from app.constants.system import _SYSTEM

# utils
from app.agents.sql_agent.utils.schema_profile import schema_text_enriched, dataset_tables
from app.agents.sql_agent.utils.table_hints import likely_table_hint
from app.agents.sql_agent.utils.question_refine import refine_question
from app.agents.sql_agent.utils.sqlextract import extract_sql
from app.agents.sql_agent.utils.sqlnorm import (
    normalize_to_sqlite, ensure_limit, replace_or_add_limit,
    desired_limit_from_question, force_limit_from_question,
)
from app.agents.sql_agent.utils.sqlexec import try_execute_with_repair
from app.agents.sql_agent.utils.summarize import summarize_rows

def answer_question(
    dataset: str,
    question: str,
    max_rows: int = 50,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns:
      - sql, columns, rows, row_count, raw, answer[, error]
    """
    sql_engine, _ = get_engine_for(dataset)

    def _shape(
        *, sql: Optional[str] = None, df: Optional[pd.DataFrame] = None,
        raw: Optional[str] = None, answer: Optional[str] = None, error: Optional[str] = None
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            cols: List[str] = []; rows: List[Dict[str, Any]] = []; rc = 0
        else:
            cols = list(df.columns); rows = df.to_dict(orient="records"); rc = int(df.shape[0])
        out: Dict[str, Any] = {"sql": sql, "columns": cols, "rows": rows, "row_count": rc, "raw": raw, "answer": answer}
        if error: out["error"] = error
        if error and not answer: out["answer"] = f"⚠️ {error}"
        return out

    if _SMALLTALK.match((question or "").strip()):
        return _shape(answer="💡 That doesn’t look like a database question. Please ask something about your tables.")

    # ----- Build schema + profiles + hint -----
    with sql_engine.begin() as conn:
        tables = dataset_tables(conn, dataset)
        if not tables:
            return _shape(error=f"No tables found for dataset '{dataset}'. Please ingest first.")
        schema_txt, profiles = schema_text_enriched(conn, dataset)

    hint = likely_table_hint(question, tables)

    # ----- Refine question -----
    refined_q, focuses = refine_question(question, profiles)

    # Helper for one SQL attempt
    def _attempt(user_q: str, *, focus_terms: List[str], label: str) -> Tuple[str, Optional[pd.DataFrame], str]:
        focus_note = f"Focus terms (from schema/examples): {', '.join(focus_terms)}\n" if focus_terms else ""
        user_prompt = (
            f"User question: {user_q}\n\n"
            f"{focus_note}"
            f"Schema:\n{schema_txt}\n\n"
            f"{hint}\n"
            "Return ONLY one valid SQLite query. No extra text.\n"
            f"{FEW_SHOTS.format(dataset=dataset)}"
        )

        raw_local = _ollama_chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_prompt}],
            temperature=0.0, model=model,
        )
        sql_local = extract_sql(raw_local or "")
        if not sql_local:
            return "", None, f"[{label}] No SQL extracted."

        sql_local = sql_local.replace("{dataset}", dataset)

        # Normalize + limit
        sql_norm, explicit_n = normalize_to_sqlite(sql_local)
        override_n = force_limit_from_question(user_q, max_rows)
        if override_n is not None:
            final_sql_local = replace_or_add_limit(sql_norm, override_n)
        else:
            n_from_q = desired_limit_from_question(user_q, max_rows)
            final_sql_local = ensure_limit(sql_norm, explicit_n or n_from_q or max_rows)

        # Execute
        try:
            with sql_engine.begin() as conn2:
                df_local = try_execute_with_repair(conn2, final_sql_local)
        except Exception as e:
            return final_sql_local, None, f"[{label}] SQL execution failed: {e}"

        return final_sql_local, df_local, f"[{label}] OK ({int(df_local.shape[0])} rows)."

    # Try refined first, then original if needed
    sql_final = None
    df_final: Optional[pd.DataFrame] = None
    attempt_logs: List[str] = []

    sql1, df1, log1 = _attempt(refined_q, focus_terms=focuses, label="refined")
    attempt_logs.append(log1)

    if df1 is not None and df1.shape[0] > 0:
        sql_final, df_final = sql1, df1
        used_q = refined_q
        used_focuses = focuses
    else:
        sql2, df2, log2 = _attempt(question, focus_terms=focuses, label="original")
        attempt_logs.append(log2)
        sql_final, df_final = sql2, df2
        used_q = question
        used_focuses = focuses

    if not sql_final:
        return _shape(raw="\n".join(attempt_logs), error="Could not extract a valid SQL query from the model output.")
    if df_final is None:
        return _shape(sql=sql_final, raw="\n".join(attempt_logs), error="SQL execution failed.")

    # ----- NL answer -----
    try:
        answer_text = nlp_answer_from_sql(
            question=used_q,
            columns=list(df_final.columns),
            rows=df_final.to_dict(orient="records"),
            model=model,
            allow_table=True,
            tone="chatty",
        )
    except Exception:
        answer_text = summarize_rows(used_q, df_final)

    debug_note = f"refined_from={question!r} -> {refined_q!r}; focus_terms={used_focuses}"
    raw_notes = "\n".join([*attempt_logs, debug_note])

    return {
        "sql": sql_final,
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records"),
        "row_count": int(df_final.shape[0]),
        "raw": raw_notes,
        "answer": answer_text,
    }
