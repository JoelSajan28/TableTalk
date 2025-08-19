import os
from sqlalchemy import create_engine

SQLITE_PATH = os.getenv("SQLITE_PATH", "./data/tabletalk.db")
os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)

engine = create_engine(f"sqlite:///{SQLITE_PATH}", future=True)
