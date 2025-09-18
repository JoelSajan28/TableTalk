from __future__ import annotations
import re
import pandas as pd

def try_execute_with_repair(conn, sql: str) -> pd.DataFrame:
    try:
        return pd.read_sql_query(sql, con=conn.connection)
    except Exception as e:
        if "no such column" in str(e).lower():
            repaired = re.sub(r'(?is)^\s*select\s+.+?\s+from', 'SELECT * FROM', sql, 1)
            return pd.read_sql_query(repaired, con=conn.connection)
        raise
