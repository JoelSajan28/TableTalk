from __future__ import annotations
from typing import Optional, Tuple
import re
from app.constants.regex_constants import _NUM_RE, _NUM_PHRASE, _FIRST_NO_NUM

def ensure_limit(sql: str, n: int) -> str:
    s = sql.rstrip().rstrip(";")
    return s if re.search(r"\bLIMIT\b", s, re.I) else f"{s} LIMIT {int(n)}"

def desired_limit_from_question(q: str, default_n: int) -> int:
    m = _NUM_RE.search(q or "")
    return int(m.group(1)) if m else int(default_n)

def normalize_to_sqlite(sql: str) -> Tuple[str, Optional[int]]:
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

def force_limit_from_question(question: str, default_max: int) -> Optional[int]:
    if m := _NUM_PHRASE.search(question or ""):
        return int(m.group(1))
    if _FIRST_NO_NUM.search(question or ""):
        return 1
    return None

def replace_or_add_limit(sql: str, n: int) -> str:
    s = sql.rstrip().rstrip(";")
    return re.sub(r'(\bLIMIT\b)\s+\d+', rf'\1 {n}', s, 1, flags=re.I) if "LIMIT" in s.upper() else f"{s} LIMIT {n}"
