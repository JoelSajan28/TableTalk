
from sqlalchemy import text

SQL_CREATE_METADATA = text("""
    CREATE TABLE IF NOT EXISTS tables_metadata (
        dataset   TEXT NOT NULL,
        name      TEXT NOT NULL,
        columns   TEXT,
        row_count INTEGER,
        UNIQUE(name)
    )
""")

SQL_CLEAR_METADATA = text("DELETE FROM tables_metadata")

SQL_UPSERT_METADATA = text("""
    INSERT OR REPLACE INTO tables_metadata (dataset, name, columns, row_count)
    VALUES (:d, :n, :c, :r)
""")