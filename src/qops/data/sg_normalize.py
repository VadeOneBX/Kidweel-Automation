"""
src/qops/data/sg_normalize.py
─────────────────────────────────────────────────────────────────────────────
Normalize raw SpotGamma export DataFrames into a clean, typed DataFrame.
Purpose:
Accept a raw pandas DataFrame as loaded from a SpotGamma history CSV.
Strip apostrophe export artifacts from numeric fields.
Normalize column names to snake_case.
Normalize trade_date to datetime.date across all SpotGamma date formats.
Cast all numeric columns to float64.
Return a clean DataFrame sorted ascending by trade_date.
No Redis. No ranking. No execution. No Alpaca calls.
Dependencies:
pandas
Assumptions:
Input DataFrame was read with dtype=str (no pre-casting).
CSV was exported from SpotGamma history table.
Date column is named "Trade Date" in the raw export.
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import re
from datetime import date
from typing import Final

import pandas as pd

# ─── Column name mapping: raw SpotGamma → snake_case ─────────────────────────
COLUMN_MAP: Final[dict[str, str]] = {
    "Trade Date": "trade_date",
    "Previous Close": "previous_close",
    "Low Vol Point": "low_vol_point",
    "High Vol Point": "high_vol_point",
    "Call Gamma": "call_gamma",
    "Put Gamma": "put_gamma",
    "Call Delta": "call_delta",
    "Put Delta": "put_delta",
    "Next Exp Gamma": "next_exp_gamma",
    "Next Exp Delta": "next_exp_delta",
    "Top Gamma Exp": "top_gamma_exp",
    "Top Delta Exp": "top_delta_exp",
    "Call Volume": "call_volume",
    "Put Volume": "put_volume",
    "Next Exp Call Vol": "next_exp_call_vol",
    "Next Exp Put Vol": "next_exp_put_vol",
    "Put/Call OI Ratio": "put_call_oi_ratio",
    "Gamma Ratio": "gamma_ratio",
    "Delta Ratio": "delta_ratio",
    "NE Skew": "ne_skew",
    "Skew": "skew",
    "1 M RV": "rv_1m",
    "1 M IV": "iv_1m",
    "IV Rank": "iv_rank",
    "Garch Rank": "garch_rank",
    "Skew Rank": "skew_rank",
    "Options Implied Move": "options_implied_move",
}

# Columns that contain expiry strings, not numeric data
_NON_NUMERIC: Final[frozenset[str]] = frozenset({
    "trade_date",
    "top_gamma_exp",
    "top_delta_exp",
})

# Supported raw date formats
_DATE_FORMATS: Final[tuple[str, ...]] = (
    "%m/%d/%y",  # M/D/YY or MM/DD/YY (SpotGamma short format: 3/9/26)
    "%Y-%m-%d",  # YYYY-MM-DD (ISO format: 2026-03-09)
    "%m/%d/%Y",  # M/D/YYYY (occasional full year)
)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw SpotGamma history DataFrame.

    Steps:
    1. Strip leading/trailing whitespace from column names.
    2. Rename columns to snake_case using COLUMN_MAP.
       Unknown columns are passed through with a generated snake_case name.
    3. Strip apostrophe export artifacts from all non-date string cells.
    4. Parse trade_date to datetime.date.
    5. Cast numeric columns to float64; non-parseable values become NaN.
    6. Sort ascending by trade_date.
    7. Reset index.

    Args:
        raw: DataFrame loaded from a SpotGamma CSV with dtype=str.

    Returns:
        Clean DataFrame with typed columns, sorted by trade_date ascending.

    Raises:
        ValueError: if the DataFrame has no rows or is missing "Trade Date".
    """
    if raw.empty:
        raise ValueError("Input DataFrame is empty.")

    df = raw.copy()

    # 1. Strip column name whitespace
    df.columns = df.columns.str.strip()
    if "Trade Date" not in df.columns:
        raise ValueError(
            "Input DataFrame must contain a 'Trade Date' column. "
            f"Got: {list(df.columns)}"
        )

    # 2. Rename known columns; convert unknown to snake_case
    df = df.rename(columns=COLUMN_MAP)
    df = df.rename(columns=_unknown_column_renames(df.columns))

    # 3. Strip apostrophe artifacts from all object columns (except trade_date)
    for col in df.select_dtypes(include="object").columns:
        if col == "trade_date":
            continue
        df[col] = df[col].str.lstrip("'").str.strip()

    # 4. Parse trade_date
    df["trade_date"] = df["trade_date"].apply(_parse_date)

    # 5. Cast numeric columns
    numeric_cols = [c for c in df.columns if c not in _NON_NUMERIC]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 6 & 7. Sort and reset
    df = df.sort_values("trade_date", ascending=True).reset_index(drop=True)
    return df


def _parse_date(raw_val: str | date) -> date:
    """
    Parse a SpotGamma date string to datetime.date.

    Handles:
    M/D/YY → e.g. "3/9/26"
    MM/DD/YY → e.g. "03/09/26"
    YYYY-MM-DD → e.g. "2026-03-09"
    M/D/YYYY → e.g. "3/9/2026"

    Returns datetime.date. Returns the original value if already a date.
    Raises ValueError on unparseable input.
    """
    if isinstance(raw_val, date):
        return raw_val
    if isinstance(raw_val, str):
        val = raw_val.strip().lstrip("'")
        for fmt in _DATE_FORMATS:
            try:
                return pd.to_datetime(val, format=fmt).date()
            except (ValueError, TypeError):
                continue
        try:
            return pd.to_datetime(val).date()
        except (ValueError, TypeError):
            pass
        raise ValueError(f"Cannot parse date value: {raw_val!r}")
    raise TypeError(f"Expected str or date, got {type(raw_val).__name__}")


def _unknown_column_renames(columns: pd.Index) -> dict[str, str]:
    """
    Build a rename dict for columns not already covered by COLUMN_MAP values.

    Converts remaining title-case / mixed-case column names to snake_case.
    """
    mapped_values = set(COLUMN_MAP.values())
    rename: dict[str, str] = {}
    for col in columns:
        col_str = str(col)
        if col_str not in mapped_values:
            snake = _snake(col_str)
            if snake != col_str:
                rename[col_str] = snake
    return rename


def _snake(name: str) -> str:
    """Convert a column name string to snake_case."""
    s = name.strip()
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower().strip("_")
