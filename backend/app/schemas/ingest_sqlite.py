from typing import List
from pydantic import BaseModel

class TableInfo(BaseModel):
    name: str
    columns: List[str]
    rows: int

class IngestSqliteResponse(BaseModel):
    filename: str
    sqlite_path: str
    dataset: str
    tables: List[TableInfo]
