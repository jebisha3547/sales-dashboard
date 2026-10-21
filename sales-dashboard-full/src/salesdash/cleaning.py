"""Ingestion and cleaning of raw sales data.

Implements the data quality rules from the project document:
  * standardise column names (also maps the public Superstore headers)
  * validate required columns with a clear error (TC-02)
  * remove exact duplicates and rows with missing dates/amounts
  * separate returns (negative quantity/amount)
  * fill missing optional values
  * flag (or cap) outliers using the IQR rule
  * produce a cleaning report showing what was fixed
"""
from __future__ import annotations

import io
import re
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["order_id", "order_date", "product_id", "category", "quantity", "sales_amount"]
OPTIONAL_TEXT = [
    "product_name", "sub_category", "customer_id", "customer_segment", "country",
    "region", "state", "city", "store_id", "sales_channel", "shipping_mode",
]
OPTIONAL_NUMERIC = ["unit_price", "discount", "cost", "profit", "promotion_flag"]
TITLE_CASE_COLUMNS = ["category", "sub_category", "region", "state", "city", "customer_segment"]

# Normalised source header -> project column name (covers the public Superstore files)
COLUMN_ALIASES = {
    "sales": "sales_amount",
    "revenue": "sales_amount",
    "segment": "customer_segment",
    "ship_mode": "shipping_mode",
    "channel": "sales_channel",
}


class MissingColumnsError(ValueError):
    """Raised when required columns are absent. Names every missing column."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Missing required column(s): {', '.join(missing)}")


@dataclass
class CleaningReport:
    rows_in: int = 0
    rows_out: int = 0
    duplicates_removed: int = 0
    invalid_rows_dropped: int = 0
    returns_separated: int = 0
    outliers_flagged: int = 0
    outlier_action: str = "flag"
    unit_price_derived: int = 0
    columns_renamed: dict = field(default_factory=dict)
    optional_columns_added: list = field(default_factory=list)
    missing_values_filled: dict = field(default_factory=dict)
    date_min: str = ""
    date_max: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Rows in:               {self.rows_in:,}",
            f"Duplicates removed:    {self.duplicates_removed:,}",
            f"Invalid rows dropped:  {self.invalid_rows_dropped:,}  (missing date/amount/key fields)",
            f"Returns separated:     {self.returns_separated:,}",
            f"Rows out (sales):      {self.rows_out:,}",
            f"Outliers ({self.outlier_action}):".ljust(23) + f"{self.outliers_flagged:,}",
            f"Unit price derived:    {self.unit_price_derived:,}",
            f"Date range:            {self.date_min} to {self.date_max}",
        ]
        if self.columns_renamed:
            lines.append(f"Columns renamed:       {self.columns_renamed}")
        if self.optional_columns_added:
            lines.append(f"Optional cols added:   {self.optional_columns_added}")
        if self.missing_values_filled:
            lines.append(f"Missing values filled: {self.missing_values_filled}")
        return "\n".join(lines)


@dataclass
class CleaningResult:
    sales: pd.DataFrame
    returns: pd.DataFrame
    report: CleaningReport


def normalise_name(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


def standardise_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Rename columns to snake_case project names. Returns (df, {old: new} for changed)."""
    mapping, new_cols = {}, []
    for col in df.columns:
        norm = normalise_name(col)
        new = COLUMN_ALIASES.get(norm, norm)
        if new != col:
            mapping[col] = new
        new_cols.append(new)
    out = df.copy()
    out.columns = new_cols
    out = out.loc[:, ~out.columns.duplicated()]
    return out, mapping


def validate_required(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MissingColumnsError(missing)


def _is_text(series: pd.Series) -> bool:
    return pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series)


def clean_sales_data(
    raw: pd.DataFrame,
    dayfirst: bool = False,
    outlier_action: str = "flag",
    iqr_k: float = 1.5,
) -> CleaningResult:
    """Clean a raw sales table.

    outlier_action: "flag" (default, adds is_outlier), "cap" (clip sales_amount to IQR
    bounds within each category) or "keep" (no outlier handling).
    """
    if outlier_action not in {"flag", "cap", "keep"}:
        raise ValueError("outlier_action must be 'flag', 'cap' or 'keep'")

    report = CleaningReport(rows_in=len(raw), outlier_action=outlier_action)
    df, report.columns_renamed = standardise_columns(raw)
    validate_required(df)

    # trim whitespace in text columns
    for col in df.columns:
        if _is_text(df[col]):
            df[col] = df[col].astype("string").str.strip()
            df[col] = df[col].replace("", pd.NA)

    # exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    report.duplicates_removed = before - len(df)

    # types
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=dayfirst)
    if "ship_date" in df.columns:
        df["ship_date"] = pd.to_datetime(df["ship_date"], errors="coerce", dayfirst=dayfirst)
    for col in ["quantity", "sales_amount"] + [c for c in OPTIONAL_NUMERIC if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # rows missing key fields
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLUMNS).copy()
    report.invalid_rows_dropped = before - len(df)

    # tidy category-like text
    for col in TITLE_CASE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.title()
    df["order_date"] = df["order_date"].dt.normalize()

    # optional columns: add if absent, fill if partly missing
    for col in OPTIONAL_TEXT:
        if col not in df.columns:
            df[col] = pd.NA
            report.optional_columns_added.append(col)
        n_missing = int(df[col].isna().sum())
        if n_missing and col not in report.optional_columns_added:
            report.missing_values_filled[col] = n_missing
        df[col] = df[col].astype("string").fillna("Unknown")
    for col, default in (("discount", 0.0), ("promotion_flag", 0)):
        if col not in df.columns:
            df[col] = default
            report.optional_columns_added.append(col)
        else:
            n_missing = int(df[col].isna().sum())
            if n_missing:
                report.missing_values_filled[col] = n_missing
            df[col] = df[col].fillna(default)
    for col in ("cost", "profit", "unit_price"):
        if col not in df.columns:
            df[col] = np.nan
            if col != "unit_price":
                report.optional_columns_added.append(col)

    # unit_price: derive from sales / (quantity * (1 - discount)) where missing
    qty_abs = df["quantity"].abs()
    can_derive = df["unit_price"].isna() & (qty_abs > 0) & (df["discount"] < 1)
    df.loc[can_derive, "unit_price"] = (
        df.loc[can_derive, "sales_amount"].abs() / (qty_abs[can_derive] * (1 - df.loc[can_derive, "discount"]))
    )
    report.unit_price_derived = int(can_derive.sum())

    # separate returns
    is_return = (df["quantity"] < 0) | (df["sales_amount"] < 0)
    returns = df[is_return].copy()
    sales = df[~is_return].copy()
    report.returns_separated = len(returns)

    # outliers (IQR within category)
    sales["is_outlier"] = False
    returns["is_outlier"] = False
    if outlier_action != "keep" and len(sales):
        grp = sales.groupby("category")["sales_amount"]
        q1 = grp.transform(lambda s: s.quantile(0.25))
        q3 = grp.transform(lambda s: s.quantile(0.75))
        iqr = q3 - q1
        lo, hi = q1 - iqr_k * iqr, q3 + iqr_k * iqr
        mask = (sales["sales_amount"] < lo) | (sales["sales_amount"] > hi)
        report.outliers_flagged = int(mask.sum())
        sales["is_outlier"] = mask
        if outlier_action == "cap":
            sales["sales_amount"] = sales["sales_amount"].clip(lower=lo, upper=hi)

    sales = sales.sort_values("order_date", kind="stable").reset_index(drop=True)
    returns = returns.sort_values("order_date", kind="stable").reset_index(drop=True)
    report.rows_out = len(sales)
    if len(sales):
        report.date_min = sales["order_date"].min().strftime("%Y-%m-%d")
        report.date_max = sales["order_date"].max().strftime("%Y-%m-%d")
    return CleaningResult(sales=sales, returns=returns, report=report)


def read_csv_flexible(source) -> pd.DataFrame:
    """Read a CSV from a path or raw bytes; falls back to latin-1 for Windows-encoded files
    (the public Superstore CSV is one)."""
    def opener():
        return io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source

    try:
        return pd.read_csv(opener())
    except UnicodeDecodeError:
        return pd.read_csv(opener(), encoding="latin-1")


def load_and_clean_csv(path: str, **kwargs) -> CleaningResult:
    return clean_sales_data(read_csv_flexible(path), **kwargs)
