import pandas as pd
from app.services.common import _safe_name

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_safe_name(c) for c in df.columns]
    return df