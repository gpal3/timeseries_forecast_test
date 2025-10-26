"""Forecasting utilities for the SKU demand forecasting app."""
from __future__ import annotations

from typing import Literal

import pandas as pd
from pmdarima.arima import auto_arima

try:
    from prophet import Prophet

    PROPHET_INSTALLED = True
except Exception:  # pragma: no cover - guard optional dependency
    Prophet = None
    PROPHET_INSTALLED = False

ModelType = Literal["ARIMA (auto)", "Prophet"]


def _build_future_index(last_date: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    if periods <= 0:
        return pd.DatetimeIndex([], name="date")
    offset = pd.tseries.frequencies.to_offset(freq)
    start = last_date + offset
    return pd.date_range(start=start, periods=periods, freq=freq, name="date")


def _prepare_series(df_sku: pd.DataFrame, freq: str) -> pd.Series:
    series = df_sku.set_index('date')['value'].sort_index()
    full_index = pd.date_range(series.index.min(), series.index.max(), freq=freq)
    series = series.reindex(full_index, fill_value=0.0)
    series.name = 'value'
    return series


def forecast_arima(df_sku: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    series = _prepare_series(df_sku, freq)
    if series.dropna().empty or len(series.dropna()) < 3:
        raise ValueError("Need at least 3 observations for ARIMA forecasting.")

    seasonal_period = 7 if freq.upper().startswith('D') else 52
    model = auto_arima(
        series,
        seasonal=True,
        m=seasonal_period,
        error_action='ignore',
        suppress_warnings=True,
        stepwise=True,
        with_intercept=True,
    )

    forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=0.05)

    future_index = _build_future_index(series.index.max(), horizon, freq)
    forecast_df = pd.DataFrame({'date': future_index, 'yhat': forecast})
    if conf_int is not None:
        forecast_df['yhat_lo'] = conf_int[:, 0]
        forecast_df['yhat_hi'] = conf_int[:, 1]
    return forecast_df


def forecast_prophet(df_sku: pd.DataFrame, horizon: int, freq: str) -> pd.DataFrame:
    if not PROPHET_INSTALLED:
        raise ImportError("Prophet is not installed.")

    data = df_sku.rename(columns={'date': 'ds', 'value': 'y'})
    if data['y'].nunique() <= 1:
        raise ValueError("Prophet requires variability in the time series.")

    model = Prophet(interval_width=0.95, weekly_seasonality=False, daily_seasonality=False)
    if freq.upper().startswith('D'):
        model.add_seasonality(name='weekly', period=7, fourier_order=3)
    model.add_seasonality(name='yearly', period=365.25 if freq.upper().startswith('D') else 52, fourier_order=5)

    model.fit(data)

    future = model.make_future_dataframe(periods=horizon, freq=freq, include_history=False)
    forecast = model.predict(future)

    forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={'ds': 'date', 'yhat_lower': 'yhat_lo', 'yhat_upper': 'yhat_hi'})
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    return forecast_df


def forecast_dispatch(df_sku: pd.DataFrame, horizon: int, freq: str, model_type: ModelType) -> pd.DataFrame:
    if horizon <= 0:
        raise ValueError("Horizon must be a positive integer.")

    if model_type == "Prophet":
        return forecast_prophet(df_sku, horizon, freq)
    return forecast_arima(df_sku, horizon, freq)
