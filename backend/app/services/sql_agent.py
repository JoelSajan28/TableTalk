from __future__ import annotations
import os, re
from typing import Dict, Any, List, Optional
import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from app.db.sqlite import engine

# -------- Ollama client --------
import requests
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi4")

def _ollama_chat(messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": messages,
        "options": {"temperature": temperature},
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("message") or {}).get("content") or data.get("response") or ""


# -------- Safety --------
_FORBIDDEN = re.compile(
    r"\b(DELETE|UPDATE|INSERT|REPLACE|ALTER|DROP|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM|CREATE)\b",
    re.I,
)
_START_OK = re.compile(r"^\s*(SELECT|WITH|COUNT)\b", re.I)

def _is_safe(sql: str) -> bool:
    return bool(_START_OK.search(sql)) and not _FORBIDDEN.search(sql)

def _ensure_limit(sql: str, n: int) -> str:
    s = sql.rstrip().rstrip(";").rstrip()
    if re.search(r"\bLIMIT\b", s, re.I):
        return s
    return f"{s} LIMIT {int(n)}"


# -------- Schema helpers --------
def _dataset_tables(conn, dataset: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text("SELECT name, columns FROM tables_metadata WHERE dataset=:d"),
        {"d": dataset},
    ).mappings().all()
    return [
        {"name": r["name"], "columns": r["columns"].split(",") if r["columns"] else []}
        for r in rows
    ]

def _schema_text(conn, dataset: str) -> str:
    parts = []
    for t in _dataset_tables(conn, dataset):
        cols = ", ".join(t["columns"])
        parts.append(f"- {t['name']}({cols})")
    return "\n".join(parts)


# -------- LLM text → SQL helpers --------
_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.I | re.S)
_THINK_TAGS = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.I | re.S)

def _is_sql_shaped(stmt: str) -> bool:
    """
    Accept only real SQL:
      - starts with SELECT or WITH or COUNT
      - contains a FROM (for SELECT) or SELECT later if WITH
      - has a non-empty projection between SELECT and FROM (either '*' or identifiers)
    Rejects prose like: 'select from the students table ...'
    """
    s = stmt.strip().rstrip(";").strip()
    if not re.match(r"^\s*(select|with|count)\b", s, flags=re.I):
        return False

    # WITH ... must eventually have a SELECT
    if re.match(r"^\s*with\b", s, flags=re.I):
        if not re.search(r"\bselect\b", s, flags=re.I):
            return False
        # fall through; we'll let it pass

    # For SELECT, require columns before FROM
    m = re.search(r"(?is)^\s*select\s+(.+?)\s+from\b", s)
    if m:
        cols = m.group(1).strip()
        if cols == "*":
            return True
        # Must contain at least one identifier/quote instead of being empty/english
        if re.search(r'[\w"`]', cols):
            return True
        return False

    # Some WITH queries might not match the simple SELECT…FROM pattern at the top
    # but will have a SELECT later; let them pass if they contain ' from ' somewhere.
    return bool(re.search(r"(?i)\bfrom\b", s))

def _extract_sql(s: str) -> Optional[str]:
    """
    For reasoning models (e.g., deepseek-r1), strip think blocks and prose,
    then extract the last SQL-shaped SELECT/WITH statement.
    """
    if not s:
        return None

    # 0) Drop <think>...</think>
    s = _THINK_TAGS.sub("", s)

    # 1) Prefer fenced ```sql``` blocks (take the last valid one)
    blocks = _SQL_FENCE.findall(s)
    candidates: List[str] = []
    if blocks:
        for b in blocks:
            candidate = b.strip()
            # keep only up to first semicolon or end
            candidate = candidate.split(";")[0].strip()
            # some models echo examples; split off "Q:" if present
            candidate = re.split(r"\n\s*Q\s*:", candidate, maxsplit=1, flags=re.I)[0].strip()
            if _is_sql_shaped(candidate):
                candidates.append(candidate)

    # 2) If no valid fenced blocks, scan whole text for SELECT/WITH chunks
    if not candidates:
        # Split by semicolons, keep pieces that contain SELECT/WITH
        parts = re.split(r";\s*\n?", s)
        for p in parts:
            m = re.search(r"(?:^|\n)\s*(SELECT|WITH|COUNT)\b[\s\S]*$", p, flags=re.I)
            if m:
                stmt = p[m.start():].strip()
                # Drop trailing “next example” lines
                stmt = re.split(r"\n\s*Q\s*:", stmt, maxsplit=1, flags=re.I)[0].strip()
                if _is_sql_shaped(stmt):
                    candidates.append(stmt)

    if not candidates:
        return None

    # 3) Return the last valid candidate (models often place final answer last)
    return candidates[-1].rstrip(";").strip()
_is_sql_shaped
# parse "first/top N"
_NUM_RE = re.compile(r'\b(?:first|top)\s+(\d+)\b', re.I)
def _desired_limit_from_question(q: str, default_n: int) -> int:
    m = _NUM_RE.search(q or "")
    return int(m.group(1)) if m else int(default_n)

# normalize TOP/FETCH → SQLite
def _normalize_to_sqlite(sql: str) -> tuple[str, Optional[int]]:
    explicit_n = None
    s = sql

    m = re.match(r'^\s*SELECT\s+TOP\s+(\d+)\s+', s, flags=re.I)
    if m:
        explicit_n = int(m.group(1))
        s = re.sub(r'^\s*SELECT\s+TOP\s+\d+\s+', 'SELECT ', s, count=1, flags=re.I)

    m = re.search(r'\bFETCH\s+FIRST\s+(\d+)\s+ROWS\s+ONLY\b', s, flags=re.I)
    if m:
        explicit_n = int(m.group(1))
        s = re.sub(r'\bFETCH\s+FIRST\s+\d+\s+ROWS\s+ONLY\b', '', s, flags=re.I)

    return s.strip(), explicit_n

# last-resort auto-repair if model invented columns
def _try_execute_with_repair(conn, sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, con=conn.connection)
    except Exception as e:
        if "no such column" in str(e).lower():
            repaired = re.sub(r'(?is)^\s*select\s+.+?\s+from', 'SELECT * FROM', sql, count=1)
            return pd.read_sql_query(repaired, con=conn.connection)
        raise


# -------- Few-shots (no triple quotes) --------
FEW_SHOTS = (
    "Examples (use exact table names and only available columns):\n"
    "Q: first 2 customers -> SELECT * FROM {dataset}__customers ORDER BY 1 LIMIT 2\n"
    "Q: first 5 orders -> SELECT * FROM {dataset}__orders ORDER BY 1 LIMIT 5\n"
    "Q: total number of customers -> SELECT COUNT(*) AS total_customers FROM {dataset}__customers\n"
    "\n"
    "IMPORTANT: Do NOT add joins, GROUP BY, or conditions unless the question explicitly asks.\n"
)

# simple hinting
def _likely_table_hint(question: str, tables: List[Dict[str, Any]]) -> str:
    q = (question or "").lower()
    names = [t["name"] for t in tables]

    cust = next((n for n in names if "customer" in n.lower()), None)
    ords = next((n for n in names if "order" in n.lower()), None)
    stud = next((n for n in names if "student" in n.lower()), None)
    sch  = next((n for n in names if "school" in n.lower()), None)

    if "customer" in q and cust:
        return (
            f"⚠️ IMPORTANT: Always use table '{cust}' when the question is about customers. "
            f"Do not use any other table."
        )
    if "order" in q and ords:
        return f"⚠️ IMPORTANT: Always use table '{ords}' when the question is about orders."
    if "student" in q and stud:
        return f"⚠️ IMPORTANT: Always use table '{stud}' when the question is about students."
    if "school" in q and sch:
        return f"⚠️ IMPORTANT: Always use table '{sch}' when the question is about schools."
    return ""

_SYSTEM = (
    "You are a careful data analyst who writes SQLite.\n"
    "Rules:\n"
    " - Return ONLY one SQL statement.\n"
    " - NO explanations, NO Markdown, NO analysis.\n"
    " - Use EXACT table names and column names from schema.\n"
    " - If the question mentions 'customers', only use the table with 'customer' in its name.\n"
    " - If it mentions 'orders', only use the table with 'order' in its name.\n"
    " - If it mentions 'students', only use the table with 'student' in its name.\n"
    " - If it mentions 'schools', only use the table with 'school' in its name.\n"
    " - Never invent or substitute tables."
)

# detect smalltalk
_SMALLTALK = re.compile(r'^(hi|hello|hey|thanks|thank you|bye)\b', re.I)

# force-limit helpers
_FIRST_NO_NUM = re.compile(r'\bfirst\b(?!\s*\d)', re.I)
_NUM_PHRASE  = re.compile(r'\b(?:first|top)\s+(\d+)\b', re.I)

def _force_limit_from_question(question: str, default_max: int) -> Optional[int]:
    q = question or ""
    m = _NUM_PHRASE.search(q)
    if m:
        return int(m.group(1))  # first/top N
    if _FIRST_NO_NUM.search(q):
        return 1                # plain "first"
    return None                 # no override

def _replace_or_add_limit(sql: str, n: int) -> str:
    s = sql.rstrip().rstrip(";")
    if re.search(r'\bLIMIT\b\s+\d+', s, flags=re.I):
        s = re.sub(r'(\bLIMIT\b)\s+\d+', rf'\1 {n}', s, flags=re.I)
    else:
        s = f"{s} LIMIT {n}"
    return s


# -------- Main entry --------
def answer_question(
    dataset: str,
    question: str,
    max_rows: int = 50,
    model: Optional[str] = None,
    sql_engine: Engine = engine,
) -> Dict[str, Any]:

    # smalltalk → friendly error (no 400)
    if _SMALLTALK.match((question or "").strip()):
        return {
            "sql": None,
            "columns": [],
            "rows": [],
            "row_count": 0,
            "raw": None,
            "error": "💡 This is not a database question. Please ask something about your tables."
        }

    # read schema
    with sql_engine.begin() as conn:
        tables = _dataset_tables(conn, dataset)
        if not tables:
            return {"error": f"No tables found for dataset '{dataset}'. Ingest first."}
        schema_txt = _schema_text(conn, dataset)

    # build prompt
    hint = _likely_table_hint(question, tables)
    user_prompt = (
        f"User question: {question}\n\n"
        f"Schema:\n{schema_txt}\n\n"
        f"{hint}\n"
        "Return ONLY one valid SQLite query. No extra text.\n"
        f"{FEW_SHOTS.format(dataset=dataset)}"
    )
    print("USER PROMPT")
    print(user_prompt)
    # ask model
    raw = _ollama_chat(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user_prompt}],
        temperature=0.0,
    )
    print("Ollama")
    print(raw)
    # extract SQL
    sql = _extract_sql(raw or "")
    if not sql:
        return {"error": "Could not extract SQL from model response.", "raw": raw}

    # substitute dataset placeholder if the model copied examples
    sql = sql.replace("{dataset}", dataset)

    # safety
    if not _is_safe(sql):
        return {"error": "Only read-only SELECT/WITH queries are allowed.", "sql": sql, "raw": raw}

    # normalize dialect (TOP/FETCH) and compute final limit
    sql_norm, explicit_n = _normalize_to_sqlite(sql)

    # hard override from NL: "first" / "first N"
    override_n = _force_limit_from_question(question, max_rows)
    if override_n is not None:
        final_sql = _replace_or_add_limit(sql_norm, override_n)
    else:
        n_from_q = _desired_limit_from_question(question, max_rows)
        final_n = explicit_n or n_from_q or max_rows
        final_sql = _ensure_limit(sql_norm, final_n)

    # execute
    try:
        with sql_engine.begin() as conn:
            df = _try_execute_with_repair(conn, final_sql)
            # conn.execute(text("DROP TABLE IF EXISTS demo_dataset__students"))
            # conn.execute(text("DROP TABLE IF EXISTS demo_dataset__schools"))
            # conn.execute(text("DELETE FROM tables_metadata WHERE name IN ('dataset__students', 'dataset__schools')"))
    except Exception as e:
        return {"error": f"SQL execution failed: {e}", "sql": final_sql, "raw": raw}

    return {
        "sql": final_sql,
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "row_count": int(df.shape[0]),
        "raw": raw,
    }
