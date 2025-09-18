from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal, TypedDict
import pandas as pd

Severity = Literal["info", "warning", "error"]

@dataclass
class TableChunk:
    dataset: str
    sheet: str
    index: int
    name: str
    df: pd.DataFrame

@dataclass
class Diagnostic:
    dataset: str
    sheet: str
    table_name: Optional[str]
    severity: Severity
    code: str
    message: str
    handled: bool
    suggestion: Optional[str] = None

class IngestTable(TypedDict):
    name: str
    columns: List[str]
    rows: int