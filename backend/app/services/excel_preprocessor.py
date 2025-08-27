# app/services/excel_preprocessor.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
import numpy as np


@dataclass
class TableChunk:
    dataset: str
    sheet: str
    index: int
    name: str
    df: pd.DataFrame


class ExcelPreprocessor:
    def __init__(
        self,
        min_table_rows: int = 3,
        gap_rows_as_split: int = 2,
        drop_all_null_cols: bool = True,
        drop_all_null_rows: bool = True,
        trim_text: bool = True,
        normalize_text_lower: bool = False,
        parse_dates: bool = True,
        infer_numeric: bool = True,
        infer_bool: bool = True,
        date_formats: Optional[List[str]] = None,
    ):
        self.min_table_rows = min_table_rows
        self.gap_rows_as_split = gap_rows_as_split
        self.drop_all_null_cols = drop_all_null_cols
        self.drop_all_null_rows = drop_all_null_rows
        self.trim_text = trim_text
        self.normalize_text_lower = normalize_text_lower
        self.parse_dates = parse_dates
        self.infer_numeric = infer_numeric
        self.infer_bool = infer_bool
        self.date_formats = date_formats or []

    # ---------- public API ----------
    def process_workbook(self, xlsx_path: Path, dataset: str) -> List[TableChunk]:
        xls = pd.ExcelFile(xlsx_path)
        out: List[TableChunk] = []
        for sheet in xls.sheet_names:
            raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, dtype=object)
            for i, df_part in enumerate(self._split_sheet_into_tables(raw)):
                df_part = self._maybe_transpose(df_part)  # NEW: detect & fix orientation
                df_part = self._fix_header_and_clean(df_part)
                if df_part.shape[0] < self.min_table_rows or df_part.shape[1] == 0:
                    continue
                df_part = self._coerce_dtypes(df_part)
                safe_sheet = self._safe_name(sheet)
                name = f"{safe_sheet}_part{i+1}" if i > 0 else safe_sheet
                out.append(TableChunk(dataset=dataset, sheet=sheet, index=i, name=name, df=df_part))
        return out

    # ---------- detect multi-table blocks ----------
    def _split_sheet_into_tables(self, raw: pd.DataFrame) -> List[pd.DataFrame]:
        raw = raw.copy()
        raw = raw.ffill(axis=1)  # fill sideways to keep merged-header cells usable
        is_blank_row = raw.isna().all(axis=1)

        blocks: List[Tuple[int, int]] = []
        start = 0
        i = 0
        while i < len(raw):
            if is_blank_row.iloc[i:i + self.gap_rows_as_split].all().all():
                end = i
                if end > start:
                    blocks.append((start, end))
                i += self.gap_rows_as_split
                start = i
            else:
                i += 1
        if start < len(raw):
            blocks.append((start, len(raw)))
        if not blocks:
            blocks = [(0, len(raw))]

        parts = []
        for a, b in blocks:
            chunk = raw.iloc[a:b].reset_index(drop=True)
            if not chunk.dropna(how="all").empty:
                parts.append(chunk)
        return parts

    # ---------- orientation detection & transpose ----------
    def _maybe_transpose(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Decide if headers are rowwise (default) or columnwise (first col). Transpose if columnwise."""
        if chunk.empty:
            return chunk

        # Score a row as header: many non-nulls, many strings, uniqueness-ish
        def _row_header_score(df: pd.DataFrame, r: int) -> float:
            row = df.iloc[r]
            non_null = row.notna().mean()
            textish = row.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
            uniq = (len(pd.Series([str(v) for v in row if pd.notna(v)]).unique()) / (len(row) or 1))
            return 0.5 * non_null + 0.35 * textish + 0.15 * uniq

        # Score a column as header: same logic but for first column
        def _col_header_score(df: pd.DataFrame, c: int) -> float:
            col = df.iloc[:, c]
            non_null = col.notna().mean()
            textish = col.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
            uniq = (len(pd.Series([str(v) for v in col if pd.notna(v)]).unique()) / (len(col) or 1))
            return 0.5 * non_null + 0.35 * textish + 0.15 * uniq

        # Try first few rows/first column
        max_check = min(5, len(chunk))
        row_scores = [ _row_header_score(chunk, r) for r in range(max_check) ]
        best_row_score = max(row_scores) if row_scores else 0.0
        first_col_score = _col_header_score(chunk, 0) if chunk.shape[1] > 0 else 0.0

        # Heuristic: if first column header score beats best row score by margin, transpose
        if first_col_score >= best_row_score * 1.15 and first_col_score >= 0.4:
            transposed = chunk.T.reset_index(drop=True)
            return transposed
        return chunk

    # ---------- header, cleanup ----------
    def _fix_header_and_clean(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()

        header_row = self._guess_header_row(df)
        header = df.iloc[header_row].tolist()
        header = [self._normalize_col_name(x) for x in header]
        header = self._dedupe_columns(header)

        df = df.iloc[header_row + 1:].reset_index(drop=True)
        df.columns = header

        if self.drop_all_null_cols:
            df = df.dropna(axis=1, how="all")
        if self.drop_all_null_rows:
            df = df.dropna(axis=0, how="all").reset_index(drop=True)

        if self.trim_text or self.normalize_text_lower:
            df = self._normalize_text_cells(df)

        return df

    def _guess_header_row(self, df: pd.DataFrame) -> int:
        best_row = 0
        best_score = -1.0
        limit = min(10, len(df))
        for i in range(limit):
            row = df.iloc[i]
            non_null = row.notna().sum()
            textish = sum(1 for v in row if pd.notna(v) and isinstance(v, str) and len(str(v).strip()) > 0)
            score = non_null + 0.5 * textish
            if score > best_score:
                best_score = score
                best_row = i
        return best_row

    # ---------- dtype coercion ----------
    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if self.infer_numeric:
            for c in df.columns:
                if df[c].dtype == object:
                    coerced = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="ignore")
                    if not coerced.equals(df[c]):
                        df[c] = coerced

        if self.infer_bool:
            truthy = {"true", "yes", "y", "1"}
            falsy = {"false", "no", "n", "0"}
            for c in df.columns:
                if df[c].dtype == object:
                    s = df[c].astype(str).str.strip().str.lower()
                    mask = s.isin(truthy | falsy)
                    if mask.any():
                        df.loc[mask, c] = s[mask].map(lambda x: True if x in truthy else False)

        if self.parse_dates:
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = self._coerce_dates(df[c])

        return df

    def _coerce_dates(self, s: pd.Series) -> pd.Series:
        out = s.copy()
        if self.date_formats:
            for fmt in self.date_formats:
                parsed = pd.to_datetime(out, format=fmt, errors="coerce")
                out = parsed.where(parsed.notna(), out)
        parsed_any = pd.to_datetime(out, errors="coerce", infer_datetime_format=True)
        return parsed_any.mask(parsed_any.notna(), parsed_any.dt.date).where(parsed_any.notna(), s)

    # ---------- text + names ----------
    def _normalize_text_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        def _clean(x: Any) -> Any:
            if isinstance(x, str):
                t = x.strip()
                if self.normalize_text_lower:
                    t = t.lower()
                return t
            return x
        return df.applymap(_clean)

    def _normalize_col_name(self, x: Any) -> str:
        s = str(x if x is not None else "").strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-zA-Z0-9_]", "", s)
        s = s.lower()
        if s == "" or s == "_":
            s = "col"
        return s

    def _dedupe_columns(self, cols: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        out: List[str] = []
        for c in cols:
            base = c if c else "col"
            if base not in seen:
                seen[base] = 0
                out.append(base)
            else:
                seen[base] += 1
                out.append(f"{base}_{seen[base]}")
        return out

    def _safe_name(self, x: str) -> str:
        s = str(x).strip().lower()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_]", "", s)
        s = s or "table"
        return s


if __name__ == "__main__":
    import sys
    xlsx = Path(sys.argv[1])
    dataset = sys.argv[2] if len(sys.argv) > 2 else "demo_dataset"
    prep = ExcelPreprocessor()
    tables = prep.process_workbook(xlsx, dataset)
    print(f"Found {len(tables)} tables")
    for t in tables:
        print(f"- {t.name}: {t.df.shape} cols={list(t.df.columns)[:6]}...")
