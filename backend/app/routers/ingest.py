# app/routers/ingest.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pathlib import Path
import tempfile
from sqlalchemy import text

from app.db.sqlite import get_engine_for
from app.services.excel_to_sqlite import ingest_excel_to_sqlite
from app.services.common import _safe_name
from app.schemas.ingest_sqlite import IngestSqliteResponse  # response model with diagnostics

router = APIRouter(prefix="/ingest", tags=["ingest"])

def _sanitize_diagnostics(diag: dict | None) -> dict:
    """
    Ensure diagnostics is always shaped safely for the response model:
    - fill missing dicts/lists
    - coerce None -> "" for optional string-ish fields when present
    - default severity/handled when missing
    """
    if not diag:
        return {"items": [], "summary": {}}

    items = diag.get("items") or []
    for it in items:
        # Only coerce keys if present; schema may allow Optional[str]
        for k in ("dataset", "sheet", "table_name", "code", "message", "suggestion"):
            if k in it and it.get(k) is None:
                it[k] = ""
        if it.get("severity") is None:
            it["severity"] = "info"
        if it.get("handled") is None:
            it["handled"] = False

    return {
        "items": items,
        "summary": diag.get("summary") or {}
    }

@router.get("/tables")
def list_tables(
    dataset: str = Query(..., description="Dataset name (prefix for tables)")
):
    """
    List registered tables for a given dataset from tables_metadata.
    """
    ds_key = _safe_name(dataset)
    engine, _ = get_engine_for(ds_key)
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
                SELECT dataset, name, columns, row_count
                FROM tables_metadata
                WHERE dataset = :d
                ORDER BY name
            """),
            {"d": ds_key},
        ).mappings().all()
    return {"tables": [dict(r) for r in rows]}

@router.post(
    "/excel",
    summary="Upload an Excel file and load each worksheet into SQLite",
    response_model=IngestSqliteResponse,
)
async def ingest_excel(
    file: UploadFile = File(..., description="Excel file (.xlsx/.xls)"),
    dataset: str = Form(..., description="Dataset name (prefix for tables)"),
):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=415, detail="Please upload an .xlsx or .xls file.")

    tmp_path: Path | None = None
    try:
        # Persist upload to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        # Ingest (returns tables + diagnostics)
        result = ingest_excel_to_sqlite(tmp_path, dataset=dataset)

        return {
            "filename": file.filename,
            "sqlite_path": result["sqlite_path"],
            "dataset": result["dataset"],
            "tables": result["tables"],
            "diagnostics": _sanitize_diagnostics(result.get("diagnostics")),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest failed: {e}")

    finally:
        if tmp_path:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
