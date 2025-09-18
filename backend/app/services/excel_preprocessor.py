# app/services/excel_preprocessor.py
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import pandas as pd


# ------------------------- Data Model ------------------------- #

@dataclass
class TableChunk:
    dataset: str
    sheet: str
    index: int
    name: str
    df: pd.DataFrame


Severity = Literal["info", "warning", "error"]

@dataclass
class Diagnostic:
    dataset: str
    sheet: str
    table_name: str | None
    severity: Severity
    code: str
    message: str
    handled: bool
    suggestion: str | None = None


# ------------------------- Preprocessor ------------------------- #

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
        *,
        # NEW knobs
        fill_merged_down: bool = True,
        single_value_row_is_separator: bool = True,
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

        # New behavior
        self.fill_merged_down = fill_merged_down
        self.single_value_row_is_separator = single_value_row_is_separator

        self.diagnostics: List[Diagnostic] = []

    # --------------------- Diagnostics helpers --------------------- #

    def _emit_diag(
        self,
        *,
        dataset: str,
        sheet: str,
        table_name: str | None,
        severity: Severity,
        code: str,
        message: str,
        handled: bool,
        suggestion: str | None = None,
    ) -> None:
        self.diagnostics.append(
            Diagnostic(
                dataset=dataset,
                sheet=sheet,
                table_name=table_name,
                severity=severity,
                code=code,
                message=message,
                handled=handled,
                suggestion=suggestion,
            )
        )

    def _is_vague_sheet_name(self, name: str) -> bool:
        return re.fullmatch(r"(sheet|sheet\d+|sheet_\d+|table|data)\s*\d*", name.strip(), re.I) is not None

    def _looks_like_vertical_headers(self, df: pd.DataFrame) -> bool:
        if df.empty or df.shape[1] == 0:
            return False
        first_col = df.iloc[:, 0]
        strings_ratio = first_col.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
        top_row_nulls = df.iloc[0].isna().mean()
        return strings_ratio >= 0.7 and top_row_nulls >= 0.5

    def _mixed_type_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Flag only TRUE mixed-type columns (or object columns with a mix of numeric-looking and non-numeric strings).
        If all values are numeric-looking strings, do NOT warn.
        """
        out: Dict[str, str] = {}
        for c in df.columns:
            series = df[c].dropna()
            if series.empty:
                continue

            sample = series.head(500)
            types = set(type(x).__name__ for x in sample)

            if len(types) > 1:
                # heterogeneous Python types present
                out[c] = ", ".join(sorted(types))
            else:
                # single dtype but might be object — check for string mix of numeric-looking & non-numeric
                if sample.dtype == object:
                    s = sample.astype(str)
                    is_num_like = s.str.match(r"^\s*-?\d+(\.\d+)?\s*$", na=False)
                    frac_num_like = is_num_like.mean()
                    # If SOME are numeric-looking but not ALL, warn about mixed content
                    if 0 < frac_num_like < 1:
                        out[c] = "text with mixed numeric-looking and non-numeric values"
                    # If all are numeric-looking strings, that's OK — no warning
        return out

    # --------------------- Public API --------------------- #

    def process_workbook(self, xlsx_path: Path, dataset: str) -> List[TableChunk]:
        """Split each sheet into tables, fix headers, coerce types, and return chunks.
        Diagnostics that may affect NL→SQL are recorded in self.diagnostics.
        """
        xls = pd.ExcelFile(xlsx_path)
        out: List[TableChunk] = []

        for sheet in xls.sheet_names:
            # Read raw cells; keep everything as object to analyze shape
            raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, dtype=object)

            if self._is_vague_sheet_name(sheet):
                self._emit_diag(
                    dataset=dataset,
                    sheet=sheet,
                    table_name=None,
                    severity="warning",
                    code="vague_sheet_name",
                    message=f"Sheet name '{sheet}' looks vague.",
                    handled=False,
                    suggestion="Rename the sheet to something descriptive (e.g., 'orders_2024').",
                )

            # Split BEFORE any filling so we don't accidentally hide separator gaps
            parts = self._split_sheet_into_tables(raw)
            if len(parts) > 1:
                self._emit_diag(
                    dataset=dataset,
                    sheet=sheet,
                    table_name=None,
                    severity="info",
                    code="multiple_tables_detected",
                    message=f"Detected {len(parts)} table-like blocks in this sheet.",
                    handled=True,
                    suggestion="Consider splitting them into separate sheets or keep clear blank row gaps.",
                )

            for i, df_part in enumerate(parts):
                original = df_part.copy()

                # Orientation check
                vertical_like = self._looks_like_vertical_headers(df_part)
                df_part2 = self._maybe_transpose(df_part)
                did_transpose = not df_part2.equals(original)
                if vertical_like:
                    self._emit_diag(
                        dataset=dataset,
                        sheet=sheet,
                        table_name=None,
                        severity="warning",
                        code="vertical_headers",
                        message="Headers appear to be vertical (first column looks like headers).",
                        handled=did_transpose,
                        suggestion="Rotate headers horizontally: one header row at the top.",
                    )
                df_part = df_part2

                # Header selection (diagnostic is based on the original block)
                header_row_guess = self._guess_header_row(original)
                if header_row_guess > 0:
                    self._emit_diag(
                        dataset=dataset,
                        sheet=sheet,
                        table_name=None,
                        severity="info",
                        code="header_offset",
                        message=f"Best header row guessed at index {header_row_guess}.",
                        handled=True,
                        suggestion="Ensure the header row is the first non-empty row when possible.",
                    )

                # Build header + clean
                df_part = self._fix_header_and_clean(df_part)

                # Drop any "single-value" rows inside the table as stray separators
                if self.single_value_row_is_separator and not df_part.empty:
                    nn = df_part.notna().sum(axis=1)
                    before = len(df_part)
                    df_part = df_part.loc[nn > 1].reset_index(drop=True)
                    removed = before - len(df_part)
                    if removed > 0:
                        self._emit_diag(
                            dataset=dataset,
                            sheet=sheet,
                            table_name=None,
                            severity="info",
                            code="single_value_rows_dropped",
                            message=f"Dropped {removed} in-table separator row(s) with only one non-null cell.",
                            handled=True,
                            suggestion="Use full rows for data; single-value rows act as section separators.",
                        )

                # NEW: Forward-fill down merged cells so labels propagate
                if self.fill_merged_down and not df_part.empty:
                    # Forward-fill down on all columns; safe after header set
                    before_na = int(df_part.isna().sum().sum())
                    # df_part = df_part.ffill(axis=0)
                    after_na = int(df_part.isna().sum().sum())
                    if after_na < before_na:
                        self._emit_diag(
                            dataset=dataset,
                            sheet=sheet,
                            table_name=None,
                            severity="info",
                            code="merged_down_fill_applied",
                            message=f"Filled {before_na - after_na} empty cell(s) by propagating merged labels downward.",
                            handled=True,
                            suggestion="Merged header/label cells were forward-filled down to rows they cover.",
                        )

                if df_part.shape[0] < self.min_table_rows or df_part.shape[1] == 0:
                    self._emit_diag(
                        dataset=dataset,
                        sheet=sheet,
                        table_name=None,
                        severity="warning",
                        code="tiny_table",
                        message="Detected a very small table (few rows/columns).",
                        handled=False,
                        suggestion="Remove stray blocks or ensure data appears under the header.",
                    )
                    continue

                # Header diagnostics
                header = list(df_part.columns)
                if len(header) != len(set(header)):
                    self._emit_diag(
                        dataset=dataset,
                        sheet=sheet,
                        table_name=None,
                        severity="warning",
                        code="duplicate_headers",
                        message="Duplicate column names detected; they were deduplicated.",
                        handled=True,
                        suggestion="Make column names unique to avoid SQL ambiguity.",
                    )

                # Type coercions + diagnostics
                df_part = self._coerce_dtypes(df_part)
                mixed = self._mixed_type_columns(df_part)
                for col, detail in mixed.items():
                    self._emit_diag(
                        dataset=dataset,
                        sheet=sheet,
                        table_name=None,
                        severity="warning",
                        code="mixed_types",
                        message=f"Column '{col}' contains mixed types: {detail}.",
                        handled=False,
                        suggestion="Normalize types (e.g., all numeric or all text).",
                    )

                sparse_cols = [c for c in df_part.columns if df_part[c].isna().mean() >= 0.6]
                for c in sparse_cols:
                    pct = int(df_part[c].isna().mean() * 100)
                    self._emit_diag(
                        dataset=dataset,
                        sheet=sheet,
                        table_name=None,
                        severity="info",
                        code="sparse_column",
                        message=f"Column '{c}' is ~{pct}% empty.",
                        handled=False,
                        suggestion="Consider removing or cleaning very sparse columns.",
                    )

                safe_sheet = self._safe_name(sheet)
                name = f"{safe_sheet}_part{i+1}" if i > 0 else safe_sheet

                out.append(TableChunk(dataset=dataset, sheet=sheet, index=i, name=name, df=df_part))

        return out

    # ------------------ Block Detection ------------------ #

    def _split_sheet_into_tables(self, raw: pd.DataFrame) -> List[pd.DataFrame]:
        """Detect multiple tables separated by gaps.
        Gap rows are:
          - entirely blank rows, OR
          - (if enabled) rows with <= 1 non-null cell (section/separator rows).
        We detect gaps BEFORE any filling so we don’t hide separators.
        """
        raw = raw.copy()

        # Identify blank rows and single-value rows
        is_blank = raw.isna().all(axis=1)
        non_null_counts = raw.notna().sum(axis=1)
        is_single_value = non_null_counts <= 1 if self.single_value_row_is_separator else pd.Series(False, index=raw.index)

        gap_mask = (is_blank | is_single_value)

        # Collect [start, end) blocks separated by runs of >= gap_rows_as_split gap rows
        blocks: List[Tuple[int, int]] = []
        start = 0
        i = 0
        while i < len(raw):
            if gap_mask.iloc[i:i + self.gap_rows_as_split].all().all():
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

        parts: List[pd.DataFrame] = []
        for a, b in blocks:
            chunk = raw.iloc[a:b].reset_index(drop=True)
            # Drop pure-blank rows that can still be inside the slice
            chunk = chunk.dropna(how="all")
            if not chunk.empty:
                parts.append(chunk.reset_index(drop=True))

        return parts

    # --------------- Orientation Detection ---------------- #

    def _maybe_transpose(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """If the first column looks like the header (column-wise), transpose."""
        if chunk.empty:
            return chunk

        def _row_header_score(df: pd.DataFrame, r: int) -> float:
            row = df.iloc[r]
            non_null = row.notna().mean()
            textish = row.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
            uniq = len(pd.Series([str(v) for v in row if pd.notna(v)]).unique()) / (len(row) or 1)
            return 0.5 * non_null + 0.35 * textish + 0.15 * uniq

        def _col_header_score(df: pd.DataFrame, c: int) -> float:
            col = df.iloc[:, c]
            non_null = col.notna().mean()
            textish = col.apply(lambda v: isinstance(v, str) and v.strip() != "").mean()
            uniq = len(pd.Series([str(v) for v in col if pd.notna(v)]).unique()) / (len(col) or 1)
            return 0.5 * non_null + 0.35 * textish + 0.15 * uniq

        max_check = min(5, len(chunk))
        row_scores = [_row_header_score(chunk, r) for r in range(max_check)]
        best_row_score = max(row_scores) if row_scores else 0.0
        first_col_score = _col_header_score(chunk, 0) if chunk.shape[1] > 0 else 0.0

        if first_col_score >= best_row_score * 1.15 and first_col_score >= 0.4:
            return chunk.T.reset_index(drop=True)
        return chunk

    # ----------------- Header & Cleaning ------------------ #

    def _fix_header_and_clean(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Choose a header row, rename columns, drop empty rows/cols, normalize text."""
        df = chunk.copy()

        header_row = self._guess_header_row(df)
        header = [self._normalize_col_name(x) for x in df.iloc[header_row].tolist()]
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
        """Pick the most header-like row among the first 10."""
        best_row = 0
        best_score = -1.0
        for i in range(min(10, len(df))):
            row = df.iloc[i]
            non_null = row.notna().sum()
            textish = sum(1 for v in row if pd.notna(v) and isinstance(v, str) and str(v).strip())
            score = non_null + 0.5 * textish
            if score > best_score:
                best_score = score
                best_row = i
        return best_row

    # ------------------ Type Coercion --------------------- #

    def _coerce_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Best-effort numeric/bool/date coercion controlled by flags."""
        df = df.copy()

        if self.infer_numeric:
            for c in df.columns:
                if df[c].dtype == object:
                    coerced = pd.to_numeric(
                        df[c].astype(str).str.replace(",", "", regex=False),
                        errors="ignore",
                    )
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
                        df.loc[mask, c] = s[mask].map(lambda x: x in truthy)

        if self.parse_dates:
            for c in df.columns:
                if df[c].dtype == object:
                    df[c] = self._coerce_dates(df[c])

        return df

    def _coerce_dates(self, s: pd.Series) -> pd.Series:
        """Try given formats first; then let pandas infer. Return date-only when parsed."""
        out = s.copy()

        if self.date_formats:
            for fmt in self.date_formats:
                parsed = pd.to_datetime(out, format=fmt, errors="coerce")
                out = parsed.where(parsed.notna(), out)

        parsed_any = pd.to_datetime(out, errors="coerce", infer_datetime_format=True)
        return parsed_any.mask(parsed_any.notna(), parsed_any.dt.date).where(parsed_any.notna(), s)

    # --------------- Text & Name Helpers ------------------ #

    def _normalize_text_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        def _clean(x: Any) -> Any:
            if isinstance(x, str):
                t = x.strip()
                return t.lower() if self.normalize_text_lower else t
            return x
        return df.applymap(_clean)

    def _normalize_col_name(self, x: Any) -> str:
        s = str(x or "").strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-zA-Z0-9_]", "", s).lower()
        return s if s and s != "_" else "col"

    def _dedupe_columns(self, cols: List[str]) -> List[str]:
        seen: Dict[str, int] = {}
        out: List[str] = []
        for c in cols:
            base = c or "col"
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
        return s or "table"


# ------------------------- CLI ------------------------- #

if __name__ == "__main__":
    import sys

    xlsx = Path(sys.argv[1])
    dataset = sys.argv[2] if len(sys.argv) > 2 else "demo_dataset"

    prep = ExcelPreprocessor()
    tables = prep.process_workbook(xlsx, dataset)

    print(f"Found {len(tables)} tables")
    for t in tables:
        print(f"- {t.name}: {t.df.shape} cols={list(t.df.columns)[:6]}...")

    if prep.diagnostics:
        print("\nDiagnostics:")
        for d in prep.diagnostics:
            print(
                f"[{d.severity}] {d.sheet}::{d.table_name or '-'} {d.code} — {d.message} "
                f"(handled={d.handled})"
                f"{' | ' + d.suggestion if d.suggestion else ''}"
            )
