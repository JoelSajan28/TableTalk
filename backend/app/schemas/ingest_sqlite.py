from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class DiagnosticItem(BaseModel):
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    dataset: Optional[str] = None
    sheet: Optional[str] = None
    table_name: Optional[str] = None  # <-- was str, make Optional[str]
    handled: bool = False
    suggestion: Optional[str] = None

class Diagnostics(BaseModel):
    items: List[DiagnosticItem] = Field(default_factory=list)
    summary: dict = Field(default_factory=dict)

class IngestSqliteResponse(BaseModel):
    filename: str
    sqlite_path: str
    dataset: str
    tables: List[dict]
    diagnostics: Optional[Diagnostics] = Field(default_factory=Diagnostics)  # allow empty
