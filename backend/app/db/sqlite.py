from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from pathlib import Path
import os

# DB path like "./data/tabularag.db" (create folder if missing)
DB_PATH = os.getenv("SQLITE_PATH", "./data/tabletalk.db")
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

engine: Engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

def execute(sql: str, **params):
    with engine.begin() as conn:
        return conn.execute(text(sql), params)
