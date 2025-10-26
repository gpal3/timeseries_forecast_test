# SKU Demand Forecasting App

This project provides a Streamlit starter application for SKU-level demand forecasting. Users can upload historical sales data, aggregate by day or week, and generate forecasts per SKU using ARIMA or Prophet (if available).

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

Prophet is optional. If it is not available, the app automatically falls back to ARIMA forecasts.

## Setup Instructions

1. **Create and activate a virtual environment (recommended).**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

2. **Install dependencies.** Install the packages defined for the app.

   ```bash
   pip install --upgrade pip
   pip install -r sku_forecast_app/requirements.txt
   ```

   - To enable Prophet-based forecasts, ensure system build tools are available and keep the `prophet` line in the requirements file. If installation fails or Prophet is not needed, comment out or remove that line.

3. **Run the Streamlit application.**

   ```bash
   streamlit run sku_forecast_app/app.py
   ```

## Usage Notes

- Upload a CSV containing at least three columns for date, SKU identifier, and sales value.
- Map the uploaded columns to the expected schema when prompted.
- Choose the aggregation frequency (daily or weekly), forecasting horizon, forecasting model, and target SKU.
- Download the generated forecast as a CSV directly from the interface.
- The starter app ships without authentication. Integrate your preferred login flow before deploying to production.

## Troubleshooting

- **Prophet installation issues**
  - Prophet may require additional system dependencies (C++ compiler, pystan). If installation is problematic, remove the Prophet requirement and rely on ARIMA forecasts.

## License

This starter project is provided without warranty. Adapt and extend it to fit your production needs.
