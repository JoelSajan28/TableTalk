# app/services/sql_agent.py
from __future__ import annotations

import re
import difflib
from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd
from sqlalchemy import text

from app.db.sqlite import get_engine_for
from app.services.ollama_client import _ollama_chat
from app.services.nl_response_agent import nlp_answer_from_sql  # Chatty NL answer
from app.constants.regex_constants import (
    _FORBIDDEN, _START_OK, _FIRST_NO_NUM, _NUM_PHRASE,
    _NUM_RE, _SMALLTALK, _SQL_FENCE, _THINK_TAGS,
)
from app.constants.few_shots import FEW_SHOTS
from app.constants.system import _SYSTEM


# ---------------- Safety & Validation ---------------- #

def _is_safe(sql: str) -> bool:
    return bool(_START_OK.search(sql)) and not _FORBIDDEN.search(sql)

def _is_sql_shaped(stmt: str) -> bool:
    s = stmt.strip().rstrip(";").strip()
    if not re.match(r"^\s*(select|with|count)\b", s, flags=re.I):
        return False
    if s.lower().startswith("with") and "select" not in s.lower():
        return False
    m = re.search(r"(?is)^\s*select\s+(.+?)\s+from\b", s)
    if m:
        cols = m.group(1).strip()
        return cols == "*" or bool(re.search(r'[\w"`]', cols))
    return " from " in s.lower()


# ---------------- Schema Utilities ---------------- #

def _dataset_tables(conn, dataset: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text("SELECT name, columns FROM tables_metadata WHERE dataset=:d"),
        {"d": dataset},
    ).mappings().all()
    return [
        {"name": r["name"], "columns": r["columns"].split(",") if r["columns"] else []}
        for r in rows
    ]


_NUMERIC_LIKE_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")

def _series_is_numeric_like_strings(s: pd.Series, threshold: float = 0.95) -> bool:
    """Return True if an object series is mostly numeric-looking strings."""
    if s.empty:
        return False
    ss = s.astype(str)
    frac = ss.str.match(_NUMERIC_LIKE_RE, na=False).mean()
    return frac >= threshold

def _choose_entity_column(candidates: List[str]) -> Optional[str]:
    """Pick a likely 'entity/name/title' column from text columns."""
    priors = ["name", "title", "task", "category", "student", "customer", "user", "account", "id"]
    for p in priors:
        for c in candidates:
            if p in c.lower():
                return c
    return candidates[0] if candidates else None

def _read_sample(conn, table: str, limit: int = 1000) -> pd.DataFrame:
    try:
        return pd.read_sql_query(f"SELECT * FROM {table} LIMIT {int(limit)}", con=conn.connection)
    except Exception:
        # Fallback if table name needs quoting or other issues
        return pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {int(limit)}', con=conn.connection)

def _profile_table(conn, table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build per-table hints:
      - text_columns: non-numeric columns with ≥50% non-null and not mostly numeric-looking strings
      - entity_column: chosen from text_columns
      - example_values: up to 20 distinct readable values from entity_column
    """
    name = table["name"]
    cols = table["columns"]
    df = _read_sample(conn, name, limit=1000)

    text_columns: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        col = df[c]
        non_null_frac = col.notna().mean()
        if non_null_frac < 0.5:
            continue

        if pd.api.types.is_numeric_dtype(col):
            continue

        if col.dtype == object and _series_is_numeric_like_strings(col.dropna()):
            continue

        if pd.api.types.is_datetime64_any_dtype(col):
            continue

        text_columns.append(c)

    entity_col = _choose_entity_column(text_columns)

    example_values: List[str] = []
    if entity_col and entity_col in df.columns:
        s = df[entity_col].dropna().astype(str).str.strip()
        # Keep reasonably "name-like" values (letters, digits, spaces, _, -, /) and length bounds
        mask = s.str.match(r"^[A-Za-z0-9][A-Za-z0-9 _\-/]{0,60}$", na=False)
        examples = s[mask].drop_duplicates().head(20).tolist()
        example_values = [e for e in examples if e]

    return {
        "name": name,
        "columns": cols,
        "text_columns": text_columns,
        "entity_column": entity_col,
        "example_values": example_values,
    }

def _schema_text_enriched(conn, dataset: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Compose a schema block that includes:
      - table(columns...)
      - text_columns: ...
      - example_values(entity_col): "a", "b", ...
    Also return the per-table profiles for refining.
    """
    lines: List[str] = []
    profiles: List[Dict[str, Any]] = []
    for t in _dataset_tables(conn, dataset):
        prof = _profile_table(conn, t)
        profiles.append(prof)
        cols = ", ".join(prof["columns"])
        lines.append(f"- {prof['name']}({cols})")
        if prof["text_columns"]:
            lines.append(f"  text_columns: {', '.join(prof['text_columns'])}")
        if prof["entity_column"] and prof["example_values"]:
            shown = ", ".join(f'"{v}"' for v in prof["example_values"])
            lines.append(f"  example_values({prof['entity_column']}): {shown}")
    return "\n".join(lines), profiles

def _schema_text(conn, dataset: str) -> str:
    txt, _ = _schema_text_enriched(conn, dataset)
    return txt


# ---------------- Query Refinement ---------------- #

_GENERIC_TAILS = (
    "implementation", "implementations", "process", "processes", "procedure", "procedures",
    "details", "overview", "report", "reports", "plan", "plans", "policy", "policies",
    "manual", "system", "deployment", "deployments", "status", "update", "updates",
)

_WORD = r"[A-Za-z0-9_\-/]+"

def _build_vocab_from_profiles(profiles: List[Dict[str, Any]]) -> Set[str]:
    vocab: Set[str] = set()
    for p in profiles:
        vocab.add(p["name"].lower())
        for c in p.get("columns", []) or []:
            vocab.add(str(c).lower())
        for ex in p.get("example_values", []) or []:
            # split on spaces to capture key tokens (e.g., "ROKS Cluster" -> "roks", "cluster")
            exl = str(ex).lower()
            vocab.add(exl)
            for tok in re.findall(_WORD, exl):
                vocab.add(tok)
    return {v for v in vocab if v and v != "-"}

def _strip_generic_tails(q: str) -> str:
    # e.g., "roks implementation" -> "roks"
    pattern = re.compile(rf"\b({_WORD})\s+({'|'.join(_GENERIC_TAILS)})\b", flags=re.I)
    # Apply repeatedly to catch chained patterns
    prev = None
    cur = q
    while prev != cur:
        prev = cur
        cur = pattern.sub(r"\1", cur)
    return cur

def _focus_terms(q: str, vocab: Set[str]) -> List[str]:
    """Pick meaningful tokens from question that match or are close to vocab entries."""
    tokens = re.findall(_WORD, q.lower())
    uniq = []
    for t in tokens:
        if t in uniq:
            continue
        # exact match
        if t in vocab:
            uniq.append(t)
            continue
        # fuzzy near-match (handles typos/casing)
        near = difflib.get_close_matches(t, vocab, n=1, cutoff=0.86)
        if near:
            uniq.append(near[0])
    return uniq[:6]  # keep it short

def _refine_question(original_q: str, profiles: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Returns (refined_question, focus_terms)
    Heuristics:
      - strip generic tails ("implementation", "overview", etc.)
      - collect focus terms by matching tokens to schema vocab / examples
      - If nothing changes, refined == original
    """
    q = (original_q or "").strip()
    q1 = _strip_generic_tails(q)
    vocab = _build_vocab_from_profiles(profiles)
    focuses = _focus_terms(q1, vocab)

    # If we stripped a tail and we have focuses, keep the reworded question;
    # otherwise fall back to original wording but still pass focus terms.
    refined = q1 if q1 != q else q
    return refined, focuses


# ---------------- SQL Extraction ---------------- #

def _extract_sql(s: str) -> Optional[str]:
    if not s:
        return None
    s = _THINK_TAGS.sub("", s)  # strip hidden thoughts if any
    candidates: List[str] = []

    for b in _SQL_FENCE.findall(s):
        candidate = b.strip().split(";")[0].strip()
        candidate = re.split(r"\n\s*Q\s*:", candidate, maxsplit=1, flags=re.I)[0].strip()
        if _is_sql_shaped(candidate):
            candidates.append(candidate)

    if not candidates:
        for p in re.split(r";\s*\n?", s):
            m = re.search(r"(?:^|\n)\s*(SELECT|WITH|COUNT)\b[\s\S]*$", p, flags=re.I)
            if m:
                stmt = p[m.start():].strip()
                stmt = re.split(r"\n\s*Q\s*:", stmt, maxsplit=1, flags=re.I)[0].strip()
                if _is_sql_shaped(stmt):
                    candidates.append(stmt)

    return candidates[-1].rstrip(";").strip() if candidates else None


# ---------------- Limits & Normalization ---------------- #

def _ensure_limit(sql: str, n: int) -> str:
    s = sql.rstrip().rstrip(";")
    return s if re.search(r"\bLIMIT\b", s, re.I) else f"{s} LIMIT {int(n)}"

def _desired_limit_from_question(q: str, default_n: int) -> int:
    m = _NUM_RE.search(q or "")
    return int(m.group(1)) if m else int(default_n)

def _normalize_to_sqlite(sql: str) -> Tuple[str, Optional[int]]:
    explicit_n = None
    s = sql

    m = re.match(r'^\s*SELECT\s+TOP\s+(\d+)\s+', s, flags=re.I)
    if m:
        explicit_n = int(m.group(1))
        s = re.sub(r'^\s*SELECT\s+TOP\s+\d+\s+', 'SELECT ', s, 1, flags=re.I)

    m = re.search(r'\bFETCH\s+FIRST\s+(\d+)\s+ROWS\s+ONLY\b', s, flags=re.I)
    if m:
        explicit_n = int(m.group(1))
        s = re.sub(r'\bFETCH\s+FIRST\s+\d+\s+ROWS\s+ONLY\b', '', s, flags=re.I)

    return s.strip(), explicit_n

def _force_limit_from_question(question: str, default_max: int) -> Optional[int]:
    if m := _NUM_PHRASE.search(question or ""):
        return int(m.group(1))
    if _FIRST_NO_NUM.search(question or ""):
        return 1
    return None

def _replace_or_add_limit(sql: str, n: int) -> str:
    s = sql.rstrip().rstrip(";")
    return re.sub(r'(\bLIMIT\b)\s+\d+', rf'\1 {n}', s, 1, flags=re.I) if "LIMIT" in s.upper() else f"{s} LIMIT {n}"


# ---------------- Repair & Execution ---------------- #

def _try_execute_with_repair(conn, sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, con=conn.connection)
    except Exception as e:
        if "no such column" in str(e).lower():
            repaired = re.sub(r'(?is)^\s*select\s+.+?\s+from', 'SELECT * FROM', sql, 1)
            return pd.read_sql_query(repaired, con=conn.connection)
        raise


# ---------------- Table Hints ---------------- #

def _likely_table_hint(question: str, tables: List[Dict[str, Any]]) -> str:
    q = (question or "").lower()
    names = [t["name"] for t in tables]

    if "customer" in q and (cust := next((n for n in names if "customer" in n), None)):
        return f"⚠️ IMPORTANT: Always use table '{cust}' for customers."
    if "order" in q and (ords := next((n for n in names if "order" in n), None)):
        return f"⚠️ IMPORTANT: Always use table '{ords}' for orders."
    if "student" in q and (stud := next((n for n in names if "student" in n), None)):
        return f"⚠️ IMPORTANT: Always use table '{stud}' for students."
    if "school" in q and (sch := next((n for n in names if "school" in n), None)):
        return f"⚠️ IMPORTANT: Always use table '{sch}' for schools."
    return ""


# ---------------- (Legacy) Row Summarization ---------------- #

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
    return ranked[0]

def _summarize_rows(question: str, df: pd.DataFrame, max_items: int = 10) -> str:
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

        bullets = []
        for _, r in top.iterrows():
            bullets.append(f"- {r.get(entity_col)} — {metric_col}: {r.get(metric_col)}")
        count = int(df.shape[0])
        lead = f"Found {count} row(s). Here are the top {len(bullets)}:"
        return "\n".join([lead, *bullets])

    cols_show = list(top.columns)[:4]
    lines = ["Found {} row(s). Sample:".format(int(df.shape[0]))]
    for _, r in top.iterrows():
        pieces = [f"{c}={r.get(c)}" for c in cols_show]
        lines.append("- " + ", ".join(pieces))
    return "\n".join(lines)


# ---------------- Main ---------------- #

def answer_question(
    dataset: str,
    question: str,
    max_rows: int = 50,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Returns:
      - sql, columns, rows, row_count, raw, answer[, error]
    Now includes:
      - query refinement (strip generic tails, find focus terms from schema)
      - two-pass SQL generation: refined first, then original if needed
    """
    sql_engine, _ = get_engine_for(dataset)

    def _shape(
        *,
        sql: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        raw: Optional[str] = None,
        answer: Optional[str] = None,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        if df is None or df.empty:
            cols: List[str] = []
            rows: List[Dict[str, Any]] = []
            rc = 0
        else:
            cols = list(df.columns)
            rows = df.to_dict(orient="records")
            rc = int(df.shape[0])

        ans = answer
        if error and not ans:
            ans = f"⚠️ {error}"

        out: Dict[str, Any] = {
            "sql": sql,
            "columns": cols,
            "rows": rows,
            "row_count": rc,
            "raw": raw,
            "answer": ans,
        }
        if error:
            out["error"] = error
        return out

    if _SMALLTALK.match((question or "").strip()):
        return _shape(
            sql=None,
            df=None,
            raw=None,
            answer="💡 That doesn’t look like a database question. Please ask something about your tables.",
        )

    # ----- Build schema + profiles + hint -----
    with sql_engine.begin() as conn:
        tables = _dataset_tables(conn, dataset)
        if not tables:
            return _shape(
                sql=None, df=None, raw=None,
                error=f"No tables found for dataset '{dataset}'. Please ingest first."
            )
        schema_txt, profiles = _schema_text_enriched(conn, dataset)

    hint = _likely_table_hint(question, tables)

    # ----- Refine question -----
    refined_q, focuses = _refine_question(question, profiles)

    # Helper for one SQL attempt
    def _attempt(user_q: str, *, focus_terms: List[str], label: str) -> Tuple[str, Optional[pd.DataFrame], str]:
        # Build a richer prompt telling the model what to focus on.
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
            temperature=0.0,
            model=model,
        )
        sql_local = _extract_sql(raw_local or "")
        if not sql_local:
            return "", None, f"[{label}] No SQL extracted."

        sql_local = sql_local.replace("{dataset}", dataset)

        # # Uncomment to enforce read-only
        # if not _is_safe(sql_local):
        #     return sql_local, None, f"[{label}] Unsafe SQL."

        # Normalize + limit
        sql_norm, explicit_n = _normalize_to_sqlite(sql_local)
        override_n = _force_limit_from_question(user_q, max_rows)
        if override_n is not None:
            final_sql_local = _replace_or_add_limit(sql_norm, override_n)
        else:
            n_from_q = _desired_limit_from_question(user_q, max_rows)
            final_sql_local = _ensure_limit(sql_norm, explicit_n or n_from_q or max_rows)

        # Execute
        try:
            with sql_engine.begin() as conn2:
                df_local = _try_execute_with_repair(conn2, final_sql_local)
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
        # fallback to original question if refined produced no rows
        sql2, df2, log2 = _attempt(question, focus_terms=focuses, label="original")
        attempt_logs.append(log2)
        sql_final, df_final = sql2, df2
        used_q = question
        used_focuses = focuses

    # If both failed to produce SQL
    if not sql_final:
        return _shape(sql=None, df=None, raw="\n".join(attempt_logs),
                      error="Could not extract a valid SQL query from the model output.")

    if df_final is None:
        return _shape(sql=sql_final, df=None, raw="\n".join(attempt_logs),
                      error="SQL execution failed.")

    # ----- ChatGPT-like NL answer from SQL rows -----
    try:
        answer_text = nlp_answer_from_sql(
            question=used_q,
            columns=list(df_final.columns),
            rows=df_final.to_dict(orient="records"),
            model=model,            # reuse same model or swap to a dedicated NL model if desired
            allow_table=True,
            tone="chatty",          # ChatGPT-like tone
        )
    except Exception:
        # Fallback to legacy summarizer if the NL agent fails
        answer_text = _summarize_rows(used_q, df_final)

    # Add a tiny debug trailer into raw to show refinement
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
