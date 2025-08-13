from typing import List
from pydantic import BaseModel, Field

class TableInfo(BaseModel):
    worksheet: str = Field(..., description="Original worksheet name")
    table: str = Field(..., description="SQLite table name created for this sheet")
    row_count: int = Field(..., description="Rows written to SQLite")
    columns: List[str] = Field(..., description="Normalized column names")

class IngestSqliteResponse(BaseModel):
    filename: str
    sqlite_path: str
    dataset: str
    tables: List[TableInfo]
