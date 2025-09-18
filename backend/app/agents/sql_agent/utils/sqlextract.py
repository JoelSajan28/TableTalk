from __future__ import annotations
from typing import Optional, List
import re
from app.constants.regex_constants import _SQL_FENCE, _THINK_TAGS
from app.agents.sql_agent.utils.sqlguard import is_sql_shaped

def extract_sql(s: str) -> Optional[str]:
    if not s:
        return None
    s = _THINK_TAGS.sub("", s)
    candidates: List[str] = []

    for b in _SQL_FENCE.findall(s):
        candidate = _clean(b)
        if is_sql_shaped(candidate):
            candidates.append(candidate)

    if not candidates:
        for p in re.split(r";\s*\n?", s):
            m = re.search(r"(?:^|\n)\s*(SELECT|WITH|COUNT)\b[\s\S]*$", p, flags=re.I)
            if m:
                stmt = _clean(p[m.start():])
                if is_sql_shaped(stmt):
                    candidates.append(stmt)

    return candidates[-1].rstrip(";").strip() if candidates else None

def _clean(block: str) -> str:
    candidate = block.strip().split(";")[0].strip()
    return re.split(r"\n\s*Q\s*:", candidate, maxsplit=1, flags=re.I)[0].strip()
