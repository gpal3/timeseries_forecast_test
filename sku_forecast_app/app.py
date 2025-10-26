"""Streamlit entry point for the SKU demand forecasting app."""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:  # pragma: no cover - environment dependency guard
    import streamlit_authenticator as stauth
    AUTH_LIB_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - executed when dependency missing
    stauth = None
    AUTH_LIB_AVAILABLE = False

from src.forecasting import PROPHET_INSTALLED, forecast_dispatch
from src.utils import aggregate_time_series, describe_series, ensure_schema, get_sku_list

ASSETS_DIR = Path(__file__).parent / "assets"


st.set_page_config(
    page_title="SKU Demand Forecasting",
    page_icon="📈",
    layout="wide",
)


def load_css() -> None:
    css_path = ASSETS_DIR / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


@st.cache_data(show_spinner=False)
def aggregate_cached(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    return aggregate_time_series(df, freq)


def _hash_passwords(passwords: List[str]) -> List[str]:
    """Hash demo passwords across streamlit-authenticator versions."""
    hasher_cls = getattr(stauth, "Hasher", None)
    if hasher_cls is None:  # pragma: no cover - defensive guard
        raise AttributeError("streamlit-authenticator missing Hasher class")

    try:
        init_params = list(inspect.signature(hasher_cls.__init__).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover - fallback when signature unavailable
        init_params = []

    # Legacy releases accept passwords via the constructor (self, passwords).
    if len(init_params) > 1:
        return hasher_cls(passwords).generate()

    # Newer releases require instantiation without arguments and passing the
    # password list to ``generate``.
    hasher = hasher_cls()
    generate_fn = getattr(hasher, "generate", None)
    if callable(generate_fn):
        return generate_fn(passwords)

    # As a last resort, attempt the legacy behaviour which will raise an
    # informative error if the API has changed again.
    return hasher_cls(passwords).generate()


def render_authentication() -> Dict[str, str]:
    if not AUTH_LIB_AVAILABLE:
        st.error(
            "Missing optional dependency `streamlit-authenticator`. "
            "Install it with `pip install streamlit-authenticator` to enable login."
        )
        return {"name": None, "status": False, "username": None}

    names = ["Demo User"]
    usernames = ["demo_user"]
    passwords = ["demo_password"]  # Replace with secrets manager in production.

    hashed_passwords = _hash_passwords(passwords)

    credentials = {
        "usernames": {
            usernames[i]: {"name": names[i], "password": hashed_passwords[i]}
            for i in range(len(usernames))
        }
    }

    authenticator = stauth.Authenticate(
        credentials,
        "sku_forecast_cookie",
        "sku_forecast_signature",
        cookie_expiry_days=1,
    )

    name, authentication_status, username = authenticator.login("Login", "main")

    if authentication_status:
        with st.sidebar:
            st.markdown(f"**Logged in as:** {name}")
            authenticator.logout("Logout", "sidebar")
    elif authentication_status is False:
        st.error("Invalid username/password")
    else:
        st.info("Please log in to continue.")

    return {
        "name": name,
        "status": authentication_status,
        "username": username,
    }


def render_metrics(df: pd.DataFrame, sku: str, freq_label: str) -> None:
    total_points, start_date, end_date = describe_series(df, sku)
    col1, col2, col3 = st.columns(3)
    col1.metric("Data points", f"{total_points}")
    col2.metric("Date range", f"{start_date.date()} → {end_date.date()}" if pd.notna(start_date) else "–")
    col3.metric("Frequency", freq_label)


def build_chart(history: pd.DataFrame, forecast: pd.DataFrame, sku: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history['date'],
            y=history['value'],
            mode='lines+markers',
            name='History',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast['date'],
            y=forecast['yhat'],
            mode='lines+markers',
            name='Forecast',
        )
    )

    if {'yhat_lo', 'yhat_hi'}.issubset(forecast.columns):
        fig.add_trace(
            go.Scatter(
                x=list(forecast['date']) + list(forecast['date'][::-1]),
                y=list(forecast['yhat_hi']) + list(forecast['yhat_lo'][::-1]),
                fill='toself',
                fillcolor='rgba(99, 110, 250, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                name='95% CI',
                showlegend=True,
            )
        )

    fig.update_layout(
        title=f"Forecast for SKU: {sku}",
        template="plotly_dark",
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Demand")
    return fig


def main() -> None:
    load_css()

    auth_state = render_authentication()
    if auth_state["status"] is not True:
        st.stop()

    st.title("SKU Demand Forecasting")
    st.caption("Upload historical sales and generate demand forecasts per SKU.")

    with st.sidebar:
        st.header("Data & Settings")
        uploaded_file = st.file_uploader("Upload CSV", type="csv")

    if not uploaded_file:
        st.info("Upload a CSV file with columns for date, sku, and value to begin.")
        return

    try:
        raw_df = load_csv(uploaded_file)
    except Exception as exc:  # pragma: no cover - defensive guard
        st.error(f"Failed to read CSV: {exc}")
        return

    if raw_df.empty:
        st.warning("The uploaded CSV is empty.")
        return

    with st.sidebar:
        st.subheader("Column Mapping")
        columns = raw_df.columns.tolist()
        date_col = st.selectbox("Date column", columns, key="date_col")
        sku_col = st.selectbox("SKU column", columns, key="sku_col")
        value_col = st.selectbox("Value column", columns, key="value_col")

    try:
        cleaned_df = ensure_schema(
            raw_df,
            {
                "date": date_col,
                "sku": sku_col,
                "value": value_col,
            },
        )
    except ValueError as err:
        st.error(str(err))
        return

    with st.sidebar:
        freq_choice = st.radio("Aggregation", ["Daily", "Weekly"], index=0)
        freq_map = {"Daily": "D", "Weekly": "W"}
        freq = freq_map[freq_choice]
        horizon = int(
            st.number_input("Forecast horizon", min_value=1, max_value=365, value=14, step=1)
        )
        model_option = st.selectbox("Model", ["ARIMA (auto)", "Prophet"], index=0)
        effective_model = model_option
        if model_option == "Prophet" and not PROPHET_INSTALLED:
            st.warning("Prophet is not installed; falling back to ARIMA (auto).")
            effective_model = "ARIMA (auto)"

    aggregated_df = aggregate_cached(cleaned_df, freq)
    if aggregated_df.empty:
        st.warning("No data available after aggregation. Check your mappings and aggregation level.")
        return

    sku_options = get_sku_list(aggregated_df)
    selected_sku = st.sidebar.selectbox("Select SKU", sku_options)

    if not selected_sku:
        st.info("Select a SKU to generate the forecast.")
        return

    sku_history = aggregated_df[aggregated_df['sku'] == selected_sku].copy()
    sku_history = sku_history.sort_values('date')

    if sku_history.empty:
        st.warning("No data points available for the selected SKU.")
        return

    render_metrics(aggregated_df, selected_sku, freq_choice)

    try:
        forecast_df = forecast_dispatch(sku_history[['date', 'value']], horizon, freq, effective_model)
    except ImportError:
        st.warning("Prophet is unavailable. Using ARIMA (auto) instead.")
        forecast_df = forecast_dispatch(sku_history[['date', 'value']], horizon, freq, "ARIMA (auto)")
    except ValueError as err:
        st.error(f"Unable to generate forecast: {err}")
        return

    forecast_df['sku'] = selected_sku

    st.plotly_chart(build_chart(sku_history, forecast_df, selected_sku), use_container_width=True)

    download_cols = ['date', 'sku', 'yhat'] + ([col for col in ['yhat_lo', 'yhat_hi'] if col in forecast_df.columns])
    download_df = forecast_df[download_cols]
    csv_bytes = download_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download forecast CSV",
        data=csv_bytes,
        file_name=f"forecast_{selected_sku}.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
