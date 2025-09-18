from __future__ import annotations
import pandas as pd

def coerce_dates(series: pd.Series, date_formats: list[str]) -> pd.Series:
    """Try given formats first; then infer. Return date-only where parsed."""
    out = series.copy()

    if date_formats:
        for fmt in date_formats:
            parsed = pd.to_datetime(out, format=fmt, errors="coerce")
            out = parsed.where(parsed.notna(), out)

    parsed_any = pd.to_datetime(out, errors="coerce", infer_datetime_format=True)
    return parsed_any.mask(parsed_any.notna(), parsed_any.dt.date).where(parsed_any.notna(), series)

def coerce_dtypes(
    df: pd.DataFrame,
    *,
    infer_numeric: bool,
    infer_bool: bool,
    parse_dates: bool,
    date_formats: list[str],
) -> pd.DataFrame:
    """Best-effort numeric/bool/date coercion controlled by flags."""
    df = df.copy()

    if infer_numeric:
        for c in df.columns:
            if df[c].dtype == object:
                coerced = pd.to_numeric(
                    df[c].astype(str).str.replace(",", "", regex=False),
                    errors="ignore",
                )
                if not coerced.equals(df[c]):
                    df[c] = coerced

    if infer_bool:
        truthy = {"true", "yes", "y", "1"}
        falsy = {"false", "no", "n", "0"}
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].astype(str).str.strip().str.lower()
                mask = s.isin(truthy | falsy)
                if mask.any():
                    df.loc[mask, c] = s[mask].map(lambda x: x in truthy)

    if parse_dates:
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = coerce_dates(df[c], date_formats)

    return df
