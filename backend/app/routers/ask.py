# app/routers/ask.py
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal

from app.agents.sql_agent.sql_agent import answer_question

router = APIRouter(prefix="/ask", tags=["ask"])

class AskRequest(BaseModel):
    dataset: str = Field(..., description="Dataset name used during ingestion")
    question: str = Field(..., description="Natural-language question")
    max_rows: int = Field(50, ge=1, le=1000)
    # Model for NL → SQL (query → SQL)
    model: Optional[str] = Field(None, description="Model for SQL generation")
    # Optional: model for SQL → NL (rows → answer). If None, backend will reuse `model`.
    nl_model: Optional[str] = Field(None, description="Model for natural-language answer")
    # Optional: answer tone; backend defaults to 'chatty' if not supplied.
    tone: Optional[Literal["chatty", "precise"]] = Field(
        None, description="Answer style: chatty (ChatGPT-like) or precise (analyst)"
    )
    # Optional: force a small markdown table when ≤ 10 rows
    allow_table: Optional[bool] = Field(
        None, description="If true, include a compact table for small results"
    )

class AskResponse(BaseModel):
    sql: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    row_count: int = 0
    raw: Optional[str] = None
    answer: Optional[str] = None
    error: Optional[str] = None

@router.post("", response_model=AskResponse, response_model_exclude_none=True)
def ask(req: AskRequest) -> AskResponse:
    dataset = (req.dataset or "").strip().lower()
    question = (req.question or "").strip()
    max_rows = int(req.max_rows or 50)

    # NOTE: The current sql_agent you pasted hard-codes tone="chatty" and
    # doesn't accept nl_model/tone/allow_table. If you extended it to accept these,
    # pass them through here. Otherwise, only pass (dataset, question, max_rows, model).
    try:
        out = answer_question(                 # if you updated sql_agent signature, use:
            dataset=dataset,
            question=question,
            max_rows=max_rows,
            model=req.model,
            # nl_model=req.nl_model,          # <- uncomment if sql_agent accepts it
            # tone=req.tone,                  # <- uncomment if sql_agent accepts it
            # allow_table=req.allow_table,    # <- uncomment if sql_agent accepts it
        ) or {}
    except TypeError:
        # Backward compatible call (original signature)
        out = answer_question(dataset, question, max_rows, model=req.model) or {}

    # ensure required fields always present
    out.setdefault("sql", None)
    out.setdefault("columns", [])
    out.setdefault("rows", [])
    out.setdefault("row_count", 0)
    out.setdefault("raw", None)
    out.setdefault("answer", None)
    out.setdefault("error", None)

    return AskResponse(**out)
