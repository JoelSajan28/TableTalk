from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

from app.services.sql_agent import answer_question

router = APIRouter(prefix="/ask", tags=["ask"])

class AskRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name used during ingestion")
    question: str = Field(..., description="Natural-language question")
    max_rows: int = Field(50, ge=1, le=1000)
    model: Optional[Literal["phi4", "deepseek-r1", "llama2-uncensored"]] = Field(
        None, description="Override Ollama model"
    )

class AskResponse(BaseModel):
    sql: Optional[str]
    columns: List[str]
    rows: List[Dict[str, Any]]
    row_count: int
    raw: Optional[str] = None

@router.post("", response_model=AskResponse)
def ask(req: AskRequest):
    out = answer_question(req.dataset.lower(), req.question, req.max_rows, model=req.model)
    if "error" in out:
        # return as 200 so Streamlit can show the error inside chat
        return AskResponse(**{
            "sql": out.get("sql"),
            "columns": out.get("columns", []),
            "rows": out.get("rows", []),
            "row_count": out.get("row_count", 0),
            "raw": out.get("raw")
        })
    return out
