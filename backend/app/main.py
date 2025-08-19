from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import ingest, ask

app = FastAPI(
    title="TableTalk API",
    version="0.2.0",
    description="Upload Excel → SQLite, then ask questions in natural language.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(ask.router)

@app.get("/healthz", tags=["meta"])
def healthz():
    return {"status": "ok"}
