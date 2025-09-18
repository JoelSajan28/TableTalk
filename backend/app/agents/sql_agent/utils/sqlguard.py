from __future__ import annotations
import re
from app.constants.regex_constants import _FORBIDDEN, _START_OK

def is_safe(sql: str) -> bool:
    """Whitelist starts + blacklist forbidden tokens."""
    return bool(_START_OK.search(sql)) and not _FORBIDDEN.search(sql)

def is_sql_shaped(stmt: str) -> bool:
    """Quick heuristic: looks like a SELECT/WITH/COUNT and has columns + FROM."""
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
