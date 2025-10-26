"""Utility helpers for the SKU demand forecasting app."""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd

REQUIRED_FIELDS = ("date", "sku", "value")


def ensure_schema(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Validate and standardise the uploaded dataframe.

    Parameters
    ----------
    df:
        Raw dataframe as uploaded by the user.
    mapping:
        Mapping from required field name (date, sku, value) to the dataframe column name.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with canonical columns: date, sku, value.

    Raises
    ------
    ValueError
        If the mapping is incomplete or results in an empty dataframe after cleaning.
    """

    missing = [field for field in REQUIRED_FIELDS if field not in mapping or not mapping[field]]
    if missing:
        raise ValueError(f"Missing column mapping for: {', '.join(missing)}")

    if len(set(mapping.values())) < len(REQUIRED_FIELDS):
        raise ValueError("Each field must map to a distinct column.")

    cleaned = df.rename(columns={mapping[field]: field for field in REQUIRED_FIELDS})

    cleaned['date'] = pd.to_datetime(cleaned['date'], errors='coerce')
    cleaned['value'] = pd.to_numeric(cleaned['value'], errors='coerce')
    cleaned = cleaned.dropna(subset=['date', 'value', 'sku'])

    if cleaned.empty:
        raise ValueError("No valid records found after cleaning. Check your mappings and data quality.")

    cleaned['sku'] = cleaned['sku'].astype(str).str.strip()
    cleaned = cleaned.sort_values('date')
    return cleaned.reset_index(drop=True)


def aggregate_time_series(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Aggregate time series to the requested frequency.

    Parameters
    ----------
    df:
        Clean dataframe with columns date, sku, value.
    freq:
        Pandas frequency string, e.g. ``'D'`` or ``'W'``.

    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with the same schema.
    """

    if df.empty:
        return df.copy()

    df_local = df.copy()
    df_local['date'] = pd.to_datetime(df_local['date'])
    df_local = df_local.set_index('date')

    aggregated = (
        df_local.groupby('sku')
        .resample(freq)
        .sum(numeric_only=True)
        .reset_index()
        .rename(columns={'value': 'value'})
    )

    aggregated = aggregated.sort_values(['sku', 'date']).reset_index(drop=True)
    return aggregated


def get_sku_list(df: pd.DataFrame) -> Iterable[str]:
    """Return sorted unique SKUs from dataframe."""

    return sorted(df['sku'].astype(str).unique()) if not df.empty else []


def describe_series(df: pd.DataFrame, sku: str) -> Tuple[int, pd.Timestamp, pd.Timestamp]:
    """Return simple metrics for the selected SKU."""

    subset = df[df['sku'] == sku]
    if subset.empty:
        return 0, pd.NaT, pd.NaT

    return len(subset), subset['date'].min(), subset['date'].max()
