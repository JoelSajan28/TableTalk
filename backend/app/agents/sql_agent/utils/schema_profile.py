from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re
import pandas as pd
from sqlalchemy import text

_NUMERIC_LIKE_RE = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")

def dataset_tables(conn, dataset: str) -> List[Dict[str, Any]]:
    rows = conn.execute(
        text("SELECT name, columns FROM tables_metadata WHERE dataset=:d"),
        {"d": dataset},
    ).mappings().all()
    return [
        {"name": r["name"], "columns": r["columns"].split(",") if r["columns"] else []}
        for r in rows
    ]

def _series_is_numeric_like_strings(s: pd.Series, threshold: float = 0.95) -> bool:
    if s.empty:
        return False
    ss = s.astype(str)
    frac = ss.str.match(_NUMERIC_LIKE_RE, na=False).mean()
    return frac >= threshold

def _choose_entity_column(candidates: List[str]) -> Optional[str]:
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
        return pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {int(limit)}', con=conn.connection)

def profile_table(conn, table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        name, columns, text_columns, entity_column, example_values
      }
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

def schema_text_enriched(conn, dataset: str) -> Tuple[str, List[Dict[str, Any]]]:
    lines: List[str] = []
    profiles: List[Dict[str, Any]] = []
    for t in dataset_tables(conn, dataset):
        prof = profile_table(conn, t)
        profiles.append(prof)
        cols = ", ".join(prof["columns"])
        lines.append(f"- {prof['name']}({cols})")
        if prof["text_columns"]:
            lines.append(f"  text_columns: {', '.join(prof['text_columns'])}")
        if prof["entity_column"] and prof["example_values"]:
            shown = ", ".join(f'"{v}"' for v in prof["example_values"])
            lines.append(f"  example_values({prof['entity_column']}): {shown}")
    return "\n".join(lines), profiles

def schema_text(conn, dataset: str) -> str:
    txt, _ = schema_text_enriched(conn, dataset)
    return txt
