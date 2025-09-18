from __future__ import annotations
from typing import Set

def unique_name(base: str, used: Set[str]) -> str:
    """Return a unique table name by suffixing _2, _3, ... if needed."""
    b = base or "table"
    name = b
    idx = 2
    while name in used:
        name = f"{b}_{idx}"
        idx += 1
    used.add(name)
    return name
