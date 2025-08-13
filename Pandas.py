import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

class ExcelSheet:
    def __init__(self, file: str | Path):
        self.file = file
        self.df: Optional[pd.DataFrame] = None

    def read_file(self, dtype: Optional[str] = None):
        """Read the first sheet into self.df"""
        self.df = pd.read_excel(self.file, dtype=dtype)

    def read_specific_sheet(self, sheet_name: str, dtype: Optional[str] = None) -> pd.DataFrame:
        """Read a specific sheet and return as DataFrame"""
        return pd.read_excel(self.file, sheet_name=sheet_name, dtype=dtype)

    def read_all_sheets(self, dtype: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """Read all sheets into a dict of DataFrames"""
        return pd.read_excel(self.file, sheet_name=None, dtype=dtype)

    def sheet_list(self) -> List[str]:
        """Return a list of sheet names"""
        xls = pd.ExcelFile(self.file)
        return xls.sheet_names


class ExcelPreprocess(ExcelSheet):
    def preprocess_columns(self):
        """Normalize column names in self.df"""
        if self.df is not None:
            self.df.columns = [c.strip().lower().replace(" ", "_") for c in self.df.columns]

    def drop_empty_rows(self):
        """Drop rows where all values are NaN"""
        if self.df is not None:
            self.df.dropna(how="all", inplace=True)

    def fill_missing(self, fill_map: Optional[Dict[str, object]] = None, default: Optional[object] = None):
        """Fill missing values"""
        if self.df is not None:
            if fill_map:
                for col, val in fill_map.items():
                    if col in self.df.columns:
                        self.df[col] = self.df[col].fillna(val)
            if default is not None:
                self.df.fillna(default, inplace=True)

    def infer_types(self, numeric_cols: Optional[List[str]] = None, datetime_cols: Optional[List[str]] = None):
        """Try to convert columns to numeric or datetime"""
        if self.df is not None:
            targets = numeric_cols or list(self.df.columns)
            for c in targets:
                try:
                    self.df[c] = pd.to_numeric(self.df[c])
                except Exception:
                    pass
            targets = datetime_cols or list(self.df.columns)
            for c in targets:
                try:
                    self.df[c] = pd.to_datetime(self.df[c])
                except Exception:
                    pass


# class DocLinker:
#     @staticmethod
#     def attach_doclinks(df: pd.DataFrame, sheet: str, base_url: str = "") -> pd.DataFrame:
#         """Add __sheet__, __row__, and doc_link columns to the DataFrame"""
#         df = df.copy()
#         df["__sheet__"] = sheet
#         df["__row__"] = (df.reset_index().index + 2).astype(int)  # Excel row numbering
#         if base_url:
#             df["doc_link"] = df["__row__"].apply(lambda r: f"{base_url}#sheet={sheet}&row={r}")
#         else:
#             df["doc_link"] = df["__row__"].apply(lambda r: f"sheet={sheet}&row={r}")
#         return df


# class PivotBuilder:
#     @staticmethod
#     def build_pivot(df: pd.DataFrame, index: str, columns: str, values: Optional[str] = None,
#                     aggfunc: str | callable = "size", fill_value: int | float = 0) -> pd.DataFrame:
#         """Create a pivot table"""
#         if values is None and aggfunc != "size":
#             raise ValueError("When values is None, aggfunc must be 'size'.")
#         if values is None:
#             pv = pd.pivot_table(df, index=index, columns=columns, aggfunc="size", fill_value=fill_value)
#         else:
#             pv = pd.pivot_table(df, index=index, columns=columns, values=values, aggfunc=aggfunc, fill_value=fill_value)
#         return pv.reset_index().rename_axis(None, axis=1)


# class DataJoiner:
#     @staticmethod
#     def merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "left",
#               suffixes: tuple = ("_x", "_y")) -> pd.DataFrame:
#         """Merge two DataFrames"""
#         return left.merge(right, on=on, how=how, suffixes=suffixes)


# Example usage
if __name__ == "__main__":
    # Read a file
    excel = ExcelPreprocess("sample_pandas.xlsx")
    excel.read_file()
    excel.preprocess_columns()
    excel.infer_types()

