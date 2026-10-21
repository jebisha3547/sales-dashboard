# Sales Analytics Dashboard with Forecasting

![CI](https://github.com/Jebisha7/sales-dashboard/actions/workflows/ci.yml/badge.svg)

An interactive sales dashboard that cleans raw transaction data, analyses it with SQL, and forecasts future sales. Several forecasting models are compared with rolling-origin backtesting, and the best one is selected automatically and shown with a prediction interval.

Built with **Python, SQL (SQLite), pandas, scikit-learn, statsmodels, Streamlit and Plotly**.

<!-- After deploying, add the live demo link here:  **Live demo:** https://your-app.streamlit.app -->

## Screenshots

| Overview | Forecast Studio |
| --- | --- |
| ![Overview](screenshots/overview.png) | ![Forecast Studio](screenshots/forecast.png) |

| Model comparison | Sales analysis |
| --- | --- |
| ![Model comparison](screenshots/model-comparison.png) | ![Sales analysis](screenshots/sales-analysis.png) |

## What it does

- **Overview:** KPI cards (revenue, profit, orders, average order value, units) with growth against the previous period of equal length, trend charts (daily to yearly), revenue by category, top products and regions.
- **Sales analysis:** breakdown by category, sub-category, region, product and more; ABC (Pareto) analysis; new vs returning customers; anomaly detection on daily or weekly revenue.
- **Forecast Studio:** forecast total sales or a single category or region, at daily, weekly or monthly level, with a configurable horizon and an 80% or 95% prediction interval. Shows the selected model, backtest error, a comparison table of all models and a downloadable forecast.
- **Data page:** upload a CSV or Excel file, map its columns to the expected fields, and see a data-quality report (duplicates, returns, missing values, outliers) with the cleaned data available for download.
- **Global filters:** date range or preset, category, region, channel and customer segment.

## Results on real data

Run on the public **Superstore** sales dataset (4 years of orders, 2014-2017), monthly total sales, 6-month horizon, 3 rolling-origin folds:

| Model | RMSE | MAPE (%) | vs seasonal naive |
| --- | --- | --- | --- |
| **Trend + seasonal regression (selected)** | 15,909 | 23.1 | 9% lower RMSE |
| Seasonal naive (baseline) | 17,487 | 23.7 | - |
| Gradient boosting | 18,829 | 29.2 | 8% higher RMSE |
| Holt-Winters | 19,349 | 26.4 | 11% higher RMSE |
| Naive | 40,886 | 74.7 | 134% higher RMSE |

What this shows, and what it does not:

- Only one of the four candidate models beat the seasonal-naive baseline, and by a modest margin. Real retail sales with 48 monthly points are noisy, and more complex models did not help.
- The comparison is based on 3 folds of 6 months (18 test points), so the 9% advantage is a weak signal and could change with other backtest windows.
- MAPE is inflated by low-sales months, where small absolute errors become large percentages. RMSE is the metric used to select the model.
- On the bundled synthetic data Holt-Winters wins easily. The ranking changes on real data, which is why models are compared by backtesting rather than chosen in advance.

## About the data

`data/superstore_synthetic.csv` is **synthetic**. I generated it with the Superstore column structure (with built-in trend, seasonality, duplicates, returns and missing values) so the app works out of the box and the cleaning logic can be tested. Do not read accuracy numbers from it. The results above come from the real Superstore dataset, which is not included in this repository: download it separately (it is publicly available, for example on Kaggle) and upload it in the app. Column names such as `Order Date`, `Sales` and `Sub-Category` are mapped automatically.

## How it works

**1. Cleaning** (`cleaning.py`) - validates required columns and names any that are missing, removes exact duplicates, drops rows missing a date or amount, separates returns (negative quantity or amount), fills missing optional fields, derives unit price when absent, and flags outliers with the IQR rule within each category. A report records what was changed.

**2. SQL layer** (`db.py`, `analytics.py`) - cleaned data is loaded into SQLite with indexes and daily, weekly and monthly summary tables. KPIs, trends, breakdowns, ABC analysis, customer analysis and anomaly detection run as SQL queries. Filter values are passed as query parameters and column names come from fixed whitelists, so filters cannot inject SQL.

**3. Forecasting** (`forecasting/`)
- Transactions are aggregated to a gap-free daily, weekly or monthly series.
- Models: naive, seasonal naive, log-linear trend with seasonal regression, gradient boosting on lag and calendar features, and Holt-Winters (via statsmodels, used when installed).
- **Backtesting:** expanding-window, rolling-origin evaluation. Each fold trains only on data before its test window, never on a random split (a test checks that no future data leaks).
- **Selection:** the model with the lowest average RMSE wins; models within 1% of the best RMSE are separated by MAPE. The app warns when no model beats the seasonal-naive baseline.
- **Prediction interval:** taken from the selected model's pooled backtest errors.

## Run it locally

```bash
git clone https://github.com/Jebisha7/sales-dashboard.git
cd sales-dashboard
pip install -r requirements-dev.txt
streamlit run app.py        # opens http://localhost:8501
pytest                      # 48 tests
python scripts/run_pipeline.py --freq M --horizon 6     # command-line version
```

Optional: create a virtual environment first (`python -m venv .venv`, then activate it). To use your own data: choose **Upload file** in the sidebar. If dates are day-first, tick the day-first box.

Docker files (`Dockerfile`, `docker-compose.yml`) are included: `docker compose up --build`.

## Project structure

```
app.py                         Streamlit dashboard (4 pages)
src/salesdash/
  cleaning.py                  validation, cleaning, quality report
  mapping.py                   column mapping for uploaded files
  db.py                        SQLite load and summary tables
  analytics.py                 KPIs, trends, breakdowns, ABC, customers, anomalies
  charts.py, styles.py         Plotly figures and CSS
  forecasting/
    series.py                  transactions -> regular time series
    models.py                  forecasting models
    backtest.py                rolling-origin backtesting, ranking, selection
    forecast.py                final forecast and prediction intervals
    metrics.py                 MAE, RMSE, MAPE, sMAPE, bias, coverage
tests/                         cleaning, analytics, forecasting, mapping, styles
scripts/run_pipeline.py        command-line pipeline
.github/workflows/ci.yml       tests run on every push
```

## Testing

48 automated tests (pytest) cover cleaning rules, KPI and filter results checked against pandas, SQL-injection protection, metric formulas on known values, backtest leakage, model selection logic, prediction-interval behaviour and column mapping. GitHub Actions runs them on every push.

## Design decisions and limitations

- **SQLite instead of PostgreSQL, Streamlit instead of a React front end** - chosen to keep the project simple and easy to run. The SQL is standard and could move to PostgreSQL.
- **Prediction intervals** use pooled backtest errors, so their width is roughly constant across the horizon. In reality uncertainty grows further ahead.
- **Forecasts assume the future resembles the past.** Promotions, product launches and external shocks are not modelled.
- **Aggregate tables** are totals only. Filtered views query the transaction table directly.
- **CSV downloads only** (no PDF or Excel export).

## Future enhancements

SARIMA and XGBoost/LightGBM models, Prophet, horizon-specific prediction intervals, more backtest folds, PDF and Excel export, user accounts with roles, a FastAPI backend with PostgreSQL, and scheduled data refresh.

## Author

GitHub: [@Jebisha7](https://github.com/Jebisha7)
