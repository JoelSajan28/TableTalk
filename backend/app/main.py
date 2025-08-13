# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ingest

openapi_tags = [
    {
        "name": "ingest",
        "description": "Upload and parse Excel files with pandas. Returns sheet summaries, columns, and previews.",
        "externalDocs": {"description": "pandas docs", "url": "https://pandas.pydata.org/docs/"}
    },
    {
        "name": "query",
        "description": "Ask questions against processed data (LLM/RAG, joins, pivots)."
    },
    {
        "name": "query",
        "description": "Ask questions against processed data (LLM/RAG, joins, pivots)."
    },
]

app = FastAPI(
    title="TableTalk API",
    version="0.1.0",
    description="Upload Excel, split into sheets, and prepare for RAG.",
    contact={"name": "Joel Sajan", "email": "joelsajan28@gmail.com"},
    license_info={"name": "IBM"},
    openapi_tags=openapi_tags,  
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(ingest.router)

@app.get("/healthz", tags=["meta"], summary="Health check")
def healthz():
    return {"status": "ok"}
