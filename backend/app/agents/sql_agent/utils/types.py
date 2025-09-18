from __future__ import annotations
from typing import List, Dict, Any, Optional, TypedDict

class TableProfile(TypedDict, total=False):
    name: str
    columns: List[str]
    text_columns: List[str]
    entity_column: Optional[str]
    example_values: List[str]

class QAResult(TypedDict, total=False):
    sql: Optional[str]
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    raw: Optional[str]
    answer: Optional[str]
    error: Optional[str]
