from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Models
from app.services.excel.models.model import TableChunk, Diagnostic, Severity

# Utils
from app.services.excel.utils.detect import is_vague_sheet_name, looks_like_vertical_headers, mixed_type_columns
from app.services.excel.utils.blocks import split_sheet_into_tables
from app.services.excel.utils.orient import maybe_transpose
from app.services.excel.utils.header import fix_header_and_clean, guess_header_row
from app.services.excel.utils.names import safe_name
from app.services.excel.utils.types_coerce import coerce_dtypes

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

    # --------------------- Public API --------------------- #

    def process_workbook(self, xlsx_path: Path, dataset: str) -> List[TableChunk]:
        """Split each sheet into tables, fix headers, coerce types, and return chunks.
        Diagnostics that may affect NL→SQL are recorded in self.diagnostics.
        """
        xls = pd.ExcelFile(xlsx_path)
        out: List[TableChunk] = []

        for sheet in xls.sheet_names:
            raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None, dtype=object)

            if is_vague_sheet_name(sheet):
                self._emit_diag(
                    dataset=dataset, sheet=sheet, table_name=None,
                    severity="warning", code="vague_sheet_name",
                    message=f"Sheet name '{sheet}' looks vague.",
                    handled=False,
                    suggestion="Rename the sheet to something descriptive (e.g., 'orders_2024').",
                )

            parts = split_sheet_into_tables(
                raw,
                gap_rows_as_split=self.gap_rows_as_split,
                single_value_row_is_separator=self.single_value_row_is_separator,
            )
            if len(parts) > 1:
                self._emit_diag(
                    dataset=dataset, sheet=sheet, table_name=None,
                    severity="info", code="multiple_tables_detected",
                    message=f"Detected {len(parts)} table-like blocks in this sheet.",
                    handled=True,
                    suggestion="Consider splitting them into separate sheets or keep clear blank row gaps.",
                )

            for i, df_part in enumerate(parts):
                original = df_part.copy()

                vertical_like = looks_like_vertical_headers(df_part)
                df_part2 = maybe_transpose(df_part)
                did_transpose = not df_part2.equals(original)
                if vertical_like:
                    self._emit_diag(
                        dataset=dataset, sheet=sheet, table_name=None,
                        severity="warning", code="vertical_headers",
                        message="Headers appear to be vertical (first column looks like headers).",
                        handled=did_transpose,
                        suggestion="Rotate headers horizontally: one header row at the top.",
                    )
                df_part = df_part2

                header_row_guess = guess_header_row(original)
                if header_row_guess > 0:
                    self._emit_diag(
                        dataset=dataset, sheet=sheet, table_name=None,
                        severity="info", code="header_offset",
                        message=f"Best header row guessed at index {header_row_guess}.",
                        handled=True,
                        suggestion="Ensure the header row is the first non-empty row when possible.",
                    )

                df_part = fix_header_and_clean(
                    df_part,
                    drop_all_null_cols=self.drop_all_null_cols,
                    drop_all_null_rows=self.drop_all_null_rows,
                    trim_text=self.trim_text,
                    normalize_text_lower=self.normalize_text_lower,
                )

                if self.single_value_row_is_separator and not df_part.empty:
                    nn = df_part.notna().sum(axis=1)
                    before = len(df_part)
                    df_part = df_part.loc[nn > 1].reset_index(drop=True)
                    removed = before - len(df_part)
                    if removed > 0:
                        self._emit_diag(
                            dataset=dataset, sheet=sheet, table_name=None,
                            severity="info", code="single_value_rows_dropped",
                            message=f"Dropped {removed} in-table separator row(s) with only one non-null cell.",
                            handled=True,
                            suggestion="Use full rows for data; single-value rows act as section separators.",
                        )

                # (Optional) Merged-down fill – currently only emits a diag; add .ffill(axis=0) if desired
                if self.fill_merged_down and not df_part.empty:
                    before_na = int(df_part.isna().sum().sum())
                    # df_part = df_part.ffill(axis=0)
                    after_na = int(df_part.isna().sum().sum())
                    if after_na < before_na:
                        self._emit_diag(
                            dataset=dataset, sheet=sheet, table_name=None,
                            severity="info", code="merged_down_fill_applied",
                            message=f"Filled {before_na - after_na} empty cell(s) by propagating merged labels downward.",
                            handled=True,
                            suggestion="Merged header/label cells were forward-filled down to rows they cover.",
                        )

                if df_part.shape[0] < self.min_table_rows or df_part.shape[1] == 0:
                    self._emit_diag(
                        dataset=dataset, sheet=sheet, table_name=None,
                        severity="warning", code="tiny_table",
                        message="Detected a very small table (few rows/columns).",
                        handled=False,
                        suggestion="Remove stray blocks or ensure data appears under the header.",
                    )
                    continue

                header = list(df_part.columns)
                if len(header) != len(set(header)):
                    self._emit_diag(
                        dataset=dataset, sheet=sheet, table_name=None,
                        severity="warning", code="duplicate_headers",
                        message="Duplicate column names detected; they were deduplicated.",
                        handled=True,
                        suggestion="Make column names unique to avoid SQL ambiguity.",
                    )

                df_part = coerce_dtypes(
                    df_part,
                    infer_numeric=self.infer_numeric,
                    infer_bool=self.infer_bool,
                    parse_dates=self.parse_dates,
                    date_formats=self.date_formats,
                )

                mixed = mixed_type_columns(df_part)
                for col, detail in mixed.items():
                    self._emit_diag(
                        dataset=dataset, sheet=sheet, table_name=None,
                        severity="warning", code="mixed_types",
                        message=f"Column '{col}' contains mixed types: {detail}.",
                        handled=False,
                        suggestion="Normalize types (e.g., all numeric or all text).",
                    )

                sparse_cols = [c for c in df_part.columns if df_part[c].isna().mean() >= 0.6]
                for c in sparse_cols:
                    pct = int(df_part[c].isna().mean() * 100)
                    self._emit_diag(
                        dataset=dataset, sheet=sheet, table_name=None,
                        severity="info", code="sparse_column",
                        message=f"Column '{c}' is ~{pct}% empty.",
                        handled=False,
                        suggestion="Consider removing or cleaning very sparse columns.",
                    )

                safe_sheet = safe_name(sheet)
                name = f"{safe_sheet}_part{i+1}" if i > 0 else safe_sheet

                out.append(TableChunk(dataset=dataset, sheet=sheet, index=i, name=name, df=df_part))

        return out

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
