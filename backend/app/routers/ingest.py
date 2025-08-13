from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import tempfile
from app.services.excel_to_sqlite import ingest_excel_to_sqlite
from app.schemas.ingest_sqlite import IngestSqliteResponse
from sqlalchemy import text
from app.db.sqlite import engine

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.get("/tables")
def list_tables():
    with engine.connect() as conn:
        res = conn.execute(text("SELECT * FROM tables_metadata")).mappings().all()
    return {"tables": res}

@router.post(
    "/excel",
    summary="Upload an Excel file and load each worksheet into SQLite",
    response_model=IngestSqliteResponse,
    description=(
        "Accepts `.xlsx`/`.xls`, reads all sheets with pandas, normalizes columns, "
        "writes one SQLite table per worksheet, and returns a summary."
    ),
    responses={
        200: {"description": "Ingested into SQLite"},
        400: {"description": "Invalid file or parse error"},
        415: {"description": "Unsupported media type"},
    },
)
async def ingest_excel(
    file: UploadFile = File(..., description="Excel file (.xlsx or .xls)"),
    dataset: str = Form(..., description="Dataset name (prefix for table names)")
):
    if not file.filename.lower().endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=415, detail="Please upload an .xlsx or .xls file.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = ingest_excel_to_sqlite(tmp_path, dataset=dataset)
        return {
            "filename": file.filename,
            "sqlite_path": "env: SQLITE_PATH or ./data/tabletalk.db",
            "dataset": result["dataset"],
            "tables": result["tables"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ingest failed: {e}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
