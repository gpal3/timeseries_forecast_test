# SKU Demand Forecasting App

This project provides a Streamlit starter application for authenticated SKU-level demand forecasting. Users can upload historical sales data, aggregate by day or week, and generate forecasts per SKU using ARIMA or Prophet (if available).

## Project Structure

```
sku_forecast_app/
├─ app.py
├─ requirements.txt
├─ assets/
│  └─ styles.css
├─ src/
│  ├─ forecasting.py
│  └─ utils.py
└─ .streamlit/
   └─ config.toml
```

## Prerequisites

- Python 3.10+
- pip

> **Note:** The app relies on the [`streamlit-authenticator`](https://pypi.org/project/streamlit-authenticator/) package for login. Ensure it is installed before running the app.

Prophet is optional. If it is not available, the app automatically falls back to ARIMA forecasts.

## Setup Instructions

1. **Create and activate a virtual environment (recommended).**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

2. **Install dependencies.** Install the packages defined for the app (includes `streamlit-authenticator`).

   ```bash
   pip install --upgrade pip
   pip install -r sku_forecast_app/requirements.txt
   ```

   - To enable Prophet-based forecasts, ensure system build tools are available and keep the `prophet` line in the requirements file. If installation fails or Prophet is not needed, comment out or remove that line.

3. **Run the Streamlit application.**

   ```bash
   streamlit run sku_forecast_app/app.py
   ```

4. **Log in with the demo credentials.**

   - Username: `demo_user`
   - Password: `demo_password`

   Replace the inline credentials configuration with a secure storage mechanism (e.g., environment variables, database, or Streamlit secrets) before deploying to production.

## Usage Notes

- Upload a CSV containing at least three columns for date, SKU identifier, and sales value.
- Map the uploaded columns to the expected schema when prompted.
- Choose the aggregation frequency (daily or weekly), forecasting horizon, forecasting model, and target SKU.
- Download the generated forecast as a CSV directly from the interface.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'streamlit_authenticator'`**
  - Ensure `streamlit-authenticator` is installed by running `pip install streamlit-authenticator`.
  - Reinstall dependencies using `pip install -r sku_forecast_app/requirements.txt`.
- **Prophet installation issues**
  - Prophet may require additional system dependencies (C++ compiler, pystan). If installation is problematic, remove the Prophet requirement and rely on ARIMA forecasts.

## License

This starter project is provided without warranty. Adapt and extend it to fit your production needs.
