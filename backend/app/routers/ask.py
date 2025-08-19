from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.sql_agent import answer_question

router = APIRouter(prefix="/ask", tags=["ask"])

class AskRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name used during ingestion")
    question: str = Field(..., description="Natural-language question")
    max_rows: int = Field(50, ge=1, le=1000)

@router.post("")
def ask(req: AskRequest):
    # Always return JSON (either with results or an "error" string). No 400s.
    return answer_question(req.dataset, req.question, req.max_rows)
