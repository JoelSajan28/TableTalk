from __future__ import annotations
from typing import Dict, List

def likely_table_hint(question: str, tables: List[Dict[str, object]]) -> str:
    q = (question or "").lower()
    names = [t["name"] for t in tables]

    def pick(substr: str, label: str) -> str:
        name = next((n for n in names if substr in n), None)
        return f"⚠️ IMPORTANT: Always use table '{name}' for {label}." if name else ""

    if "customer" in q:
        h = pick("customer", "customers")
        if h: return h
    if "order" in q:
        h = pick("order", "orders")
        if h: return h
    if "student" in q:
        h = pick("student", "students")
        if h: return h
    if "school" in q:
        h = pick("school", "schools")
        if h: return h
    return ""
