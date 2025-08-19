# app/db/sqlite.py
from __future__ import annotations
import os, re
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

SQLITE_FOLDER = Path(os.getenv("SQLITE_FOLDER", "./data"))

def _safe_dataset(dataset: str) -> str:
    # normalize: lower, spaces→underscore, strip weird chars
    s = dataset.strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]+", "_", s)

def db_path_for(dataset: str) -> Path:
    return SQLITE_FOLDER / f"{_safe_dataset(dataset)}.db"

def get_engine_for(dataset: str) -> tuple[Engine, Path]:
    path = db_path_for(dataset)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}", future=True)
    return engine, path
