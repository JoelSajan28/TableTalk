from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
import tempfile
from sqlalchemy import text
from app.db.sqlite import engine
from app.services.excel_to_sqlite import ingest_excel_to_sqlite
from app.schemas.ingest_sqlite import IngestSqliteResponse

router = APIRouter(prefix="/ingest", tags=["ingest"])

@router.get("/tables")
def list_tables():
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM tables_metadata")).mappings().all()
    return {"tables": rows}

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
