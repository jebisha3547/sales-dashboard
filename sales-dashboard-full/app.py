"""Sales Analytics Dashboard with Forecasting.   Run:  streamlit run app.py"""
from __future__ import annotations

import io
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from salesdash import analytics as A  # noqa: E402
from salesdash import charts as C  # noqa: E402
from salesdash import db  # noqa: E402
from salesdash.cleaning import REQUIRED_COLUMNS, MissingColumnsError, clean_sales_data, read_csv_flexible  # noqa: E402
from salesdash.forecasting import available_models, make_series, run_forecast  # noqa: E402
from salesdash.mapping import TARGET_FIELDS, apply_mapping, suggest_mapping  # noqa: E402
from salesdash.styles import CSS  # noqa: E402

SAMPLE_PATH = ROOT / "data" / "superstore_synthetic.csv"
TREND_FREQ = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Yearly": "Y"}
FORECAST_FREQ = {"Monthly": "M", "Weekly": "W", "Daily": "D"}
FORECAST_UNIT = {"M": "month", "W": "week", "D": "day"}
DIMENSIONS = ["category", "sub_category", "region", "state", "city", "customer_segment",
              "sales_channel", "product_name", "shipping_mode"]
PRESETS = {"Custom": None, "Last 30 days": 30, "Last 90 days": 90, "Last 12 months": 365, "All time": 0}


NAV_ICONS = {"Overview": "📊 ", "Sales analysis": "🔎 ", "Forecast Studio": "🔮 ", "Data": "🗂️ "}
MODEL_LABELS = {
    "naive": "Naive",
    "seasonal_naive": "Seasonal naive (baseline)",
    "trend_seasonal_reg": "Trend + seasonal regression",
    "gbm_trend_lags": "Gradient boosting",
    "holt_winters": "Holt-Winters",
    "prophet": "Prophet",
}
MODEL_SHORT = {   # short names for the narrow KPI card (full names are used in tables and text)
    "naive": "Naive", "seasonal_naive": "Seasonal naive", "trend_seasonal_reg": "Trend + seasonal",
    "gbm_trend_lags": "Gradient boosting", "holt_winters": "Holt-Winters", "prophet": "Prophet",
}
COLUMN_LABELS = {
    "share_pct": "Share (%)", "cum_share_pct": "Cumulative share (%)", "abc_class": "ABC class",
    "robust_z": "Robust z-score", "avg_order_value": "Avg order value",
}
# Keep the Plotly toolbar (zoom, PNG download) but drop buttons nobody needs.
PLOT_CONFIG = {"displaylogo": False, "responsive": True,
               "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"]}


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def model_label(name: str) -> str:
    return MODEL_LABELS.get(name, pretty(name))


def money(v: float) -> str:
    """Compact currency for KPI cards: $2.30M, $286.4K, $459."""
    a = abs(v)
    if a >= 1e9:
        return f"${v / 1e9:.2f}B"
    if a >= 1e6:
        return f"${v / 1e6:.2f}M"
    if a >= 1e4:
        return f"${v / 1e3:.1f}K"
    return f"${v:,.0f}"


def csv_button(label: str, df: pd.DataFrame, filename: str, key: str, primary: bool = False) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv", key=key,
                       type="primary" if primary else "secondary")


def page_header(title: str, subtitle: str = "") -> None:
    """Consistent page title with a one-line description (static text only)."""
    st.markdown(f"<div class='page-header'><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def card():
    """A bordered box that groups a chart or table."""
    return st.container(border=True)


def section(title: str, hint: str = "") -> None:
    st.markdown(f"**{title}**")
    if hint:
        st.caption(hint)


def show(fig) -> None:
    st.plotly_chart(fig, config=PLOT_CONFIG)


def table_config(df: pd.DataFrame) -> dict:
    """Readable headers and number formats for st.dataframe (display only)."""
    cfg = {}
    for col in df.columns:
        label = COLUMN_LABELS.get(col, pretty(col))
        if col in ("revenue", "profit", "avg_order_value"):
            cfg[col] = st.column_config.NumberColumn(label, format="$%.0f")
        elif col in ("units", "orders", "items", "new_customers", "returning_customers"):
            cfg[col] = st.column_config.NumberColumn(label, format="%d")
        elif col in ("share_pct", "cum_share_pct", "robust_z"):
            cfg[col] = st.column_config.NumberColumn(label, format="%.1f" if col != "robust_z" else "%.2f")
        else:
            cfg[col] = st.column_config.Column(label)
    return cfg


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def read_sample() -> pd.DataFrame:
    return read_csv_flexible(str(SAMPLE_PATH))


@st.cache_data(show_spinner=False)
def read_upload(name: str, content: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    return read_csv_flexible(content)


@st.cache_resource(show_spinner="Cleaning data and building tables...")
def prepare(source_key: str, _raw: pd.DataFrame, mapping_items: tuple, dayfirst: bool, outlier_action: str):
    df = apply_mapping(_raw, dict(mapping_items))
    result = clean_sales_data(df, dayfirst=dayfirst, outlier_action=outlier_action)
    conn = db.connect()
    db.load_transactions(conn, result.sales, result.returns)
    return result, conn


def sidebar_data():
    st.sidebar.header("Data")
    source = st.sidebar.radio("Source", ["Sample data", "Upload file"], key="source")
    if source == "Sample data":
        if not SAMPLE_PATH.exists():
            st.error(f"Sample file not found: {SAMPLE_PATH}")
            st.stop()
        raw, key = read_sample(), "sample"
        st.sidebar.caption("The bundled sample is synthetic. Upload a real file for real results.")
    else:
        upload = st.sidebar.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"])
        if upload is None:
            st.info("Upload a CSV or Excel file with your sales data, or switch to the sample data in the sidebar.")
            st.stop()
        raw, key = read_upload(upload.name, upload.getvalue()), f"{upload.name}-{upload.size}"

    suggested = suggest_mapping(list(raw.columns))
    none_option = "- not in file -"
    options = [none_option] + [str(c) for c in raw.columns]
    mapping: dict = {}
    with st.sidebar.expander("Column mapping (* required)", expanded=(source == "Upload file")):
        for field in TARGET_FIELDS:
            default = suggested[field]
            idx = options.index(str(default)) if default is not None and str(default) in options else 0
            label = field + (" *" if field in REQUIRED_COLUMNS else "")
            choice = st.selectbox(label, options, index=idx, key=f"map_{key}_{field}")
            mapping[field] = None if choice == none_option else choice
    dayfirst = st.sidebar.checkbox("Dates are day-first (dd/mm/yyyy)", value=False, key=f"dayfirst_{key}")
    outlier_action = st.sidebar.selectbox("Outliers", ["flag", "cap", "keep"], key=f"outliers_{key}",
                                          help="flag: mark them. cap: limit to IQR bounds. keep: ignore.")
    try:
        result, conn = prepare(key, raw, tuple(sorted(mapping.items())), dayfirst, outlier_action)
    except MissingColumnsError as exc:
        st.error(f"{exc}. Map them in the sidebar under 'Column mapping'.")
        st.stop()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if result.sales.empty:
        st.error("No valid sales rows remain after cleaning. Check the column mapping and date format.")
        st.stop()
    return key, raw, result, conn


def sidebar_filters(key: str, conn, sales: pd.DataFrame) -> A.Filters:
    st.sidebar.header("Filters")
    lo, hi = (date.fromisoformat(x) for x in A.date_bounds(conn))
    preset = st.sidebar.selectbox("Date range", list(PRESETS), index=4, key=f"preset_{key}")
    days = PRESETS[preset]
    if preset == "Custom":
        picked = st.sidebar.date_input("From / to", value=(lo, hi), min_value=lo, max_value=hi, key=f"dates_{key}")
        start, end = picked if isinstance(picked, (tuple, list)) and len(picked) == 2 else (lo, hi)
    elif days:
        start, end = max(lo, hi - timedelta(days=days - 1)), hi
    else:
        start, end = lo, hi
    cats = st.sidebar.multiselect("Category", sorted(sales["category"].unique()), key=f"cat_{key}")
    regions = st.sidebar.multiselect("Region", sorted(sales["region"].unique()), key=f"reg_{key}")
    channels = st.sidebar.multiselect("Channel", sorted(sales["sales_channel"].unique()), key=f"chan_{key}")
    segments = st.sidebar.multiselect("Customer segment", sorted(sales["customer_segment"].unique()), key=f"seg_{key}")
    return A.Filters(start.isoformat(), end.isoformat(), tuple(cats), tuple(regions), tuple(channels), tuple(segments))


# ---------------------------------------------------------------------- pages
def page_overview(conn, f: A.Filters) -> None:
    page_header("Overview", "Key sales metrics for the selected period and filters")
    k = A.kpis(conn, f)
    cur, growth = k["current"], k["growth_pct"] or {}
    if cur["orders"] == 0:
        st.warning("No sales match the current filters.")
        return

    def delta(name: str):
        v = growth.get(name)
        return None if v is None else f"{v:+.1f}%"

    cols = st.columns(5)
    cols[0].metric("Revenue", money(cur["revenue"]), delta("revenue"))
    cols[1].metric("Profit", money(cur["profit"]), delta("profit"))
    cols[2].metric("Orders", f"{cur['orders']:,}", delta("orders"))
    cols[3].metric("Avg order value", money(cur["avg_order_value"]), delta("avg_order_value"))
    cols[4].metric("Units sold", f"{cur['units']:,}", delta("units"))
    note = f"{k['period'][0]} to {k['period'][1]}  |  profit margin {cur['profit_margin']:.1%}  |  return rate {k['return_rate']:.1%}"
    if k["previous_period"]:
        note += f"  |  change vs {k['previous_period'][0]} to {k['previous_period'][1]}"
    st.caption(note)

    left, right = st.columns([2, 1])
    with left, card():
        c1, c2 = st.columns(2)
        freq = c1.selectbox("Trend by", list(TREND_FREQ), index=2, key="ov_freq")
        metric = c2.selectbox("Metric", ["revenue", "profit", "units", "orders"], key="ov_metric")
        trend = A.trends(conn, TREND_FREQ[freq], f)
        show(C.line_chart(trend, "period", metric, f"{pretty(metric)} - {freq.lower()}", height=350))
    with right, card():
        show(C.donut(A.breakdown(conn, "category", f), "category", "revenue", "Revenue by category", height=440))

    a, b = st.columns(2)
    with a, card():
        show(C.bar_chart(A.breakdown(conn, "product_name", f, top_n=10), "product_name", "revenue", "Top 10 products by revenue", height=400))
    with b, card():
        show(C.bar_chart(A.breakdown(conn, "region", f), "region", "revenue", "Revenue by region", height=400))


def page_analysis(conn, f: A.Filters) -> None:
    page_header("Sales analysis", "Explore sales by product, category, region and customer")
    tab_break, tab_abc, tab_cust, tab_anom = st.tabs(["Breakdown", "ABC analysis", "Customers", "Anomalies"])

    with tab_break:
        with card():
            c1, c2, c3 = st.columns(3)
            dim = c1.selectbox("Break down by", DIMENSIONS, format_func=pretty, key="bd_dim")
            metric = c2.selectbox("Metric", ["revenue", "profit", "units", "orders"], key="bd_metric")
            top_n = c3.slider("Show top", 5, 50, 15, key="bd_top")
            df = A.breakdown(conn, dim, f, metric=metric)
            show(C.bar_chart(df.head(top_n), dim, metric, f"{pretty(metric)} by {pretty(dim).lower()}"))
        with card():
            section("Data table")
            st.dataframe(df.round(2), hide_index=True, column_config=table_config(df))
            csv_button("Download table (CSV)", df, f"breakdown_{dim}.csv", "dl_breakdown")

    with tab_abc:
        with card():
            dim = st.selectbox("Classify by", ["product_name", "sub_category", "category", "city"], format_func=pretty, key="abc_dim")
            abc = A.abc_analysis(conn, f, dim)
            summary = (abc.groupby("abc_class").agg(items=(dim, "count"), revenue=("revenue", "sum"), share_pct=("share_pct", "sum"))
                       .round(1).reset_index())
            section("Summary by class", "A = items making up the first 80% of revenue, B = next 15%, C = last 5%.")
            st.dataframe(summary, hide_index=True, column_config=table_config(summary))
        with card():
            section("All items")
            st.dataframe(abc.round(2), hide_index=True, column_config=table_config(abc))
            csv_button("Download ABC table (CSV)", abc, "abc_analysis.csv", "dl_abc")

    with tab_cust:
        with card():
            nv = A.new_vs_returning(conn, f)
            show(C.stacked_bar(nv, "period", ["new_customers", "returning_customers"], "New vs returning customers per month"))
            st.caption("A customer counts as 'new' in the month of their first order in the whole dataset.")

    with tab_anom:
        with card():
            c1, c2 = st.columns(2)
            freq = c1.selectbox("Granularity", ["Daily", "Weekly"], key="an_freq")
            z = c2.slider("Sensitivity (robust z-score threshold)", 2.0, 6.0, 3.0, 0.5, key="an_z")
            an = A.anomalies(conn, TREND_FREQ[freq], f, z)
            show(C.anomaly_chart(an, f"{freq} revenue with unusual periods marked"))
        with card():
            flagged = an[an["is_anomaly"]] if "is_anomaly" in an else an.head(0)
            section(f"{len(flagged)} unusual period(s) found out of {len(an)}")
            cols = [c for c in ("period", "revenue", "robust_z") if c in flagged]
            view = flagged[cols].round(2).assign(period=lambda d: d["period"].dt.strftime("%Y-%m-%d"))
            st.dataframe(view, hide_index=True, column_config=table_config(view))


def page_forecast(key: str, sales: pd.DataFrame) -> None:
    page_header("Forecast Studio", "Compare forecasting models with backtesting and project future sales. "
                                   "Uses the full history of the chosen scope (sidebar date filters do not apply).")
    with card():
        section("Forecast settings")
        c1, c2, c3, c4 = st.columns(4)
        scope = c1.selectbox("Scope", ["Total sales", "Category", "Region"], key="fc_scope")
        column, value = None, None
        if scope != "Total sales":
            column = scope.lower()
            value = c2.selectbox(scope, sorted(sales[column].unique()), key="fc_value")
        freq_label = c3.selectbox("Frequency", list(FORECAST_FREQ), key="fc_freq")
        freq = FORECAST_FREQ[freq_label]
        default_h, max_h = {"M": (6, 24), "W": (13, 52), "D": (28, 90)}[freq]
        horizon = c4.number_input(f"Horizon ({FORECAST_UNIT[freq]}s)", 1, max_h, default_h, key=f"fc_h_{freq}")

        c5, c6 = st.columns([3, 1])
        options = available_models()
        models = c5.multiselect("Models to compare (seasonal naive baseline is always included)", options, default=options,
                                format_func=model_label, key="fc_models")
        level = c6.selectbox("Prediction interval", [0.80, 0.95], format_func=lambda x: f"{x:.0%}", key="fc_level")
        run_clicked = st.button("Run forecast", type="primary", key="fc_run")

    if run_clicked:
        data = sales if column is None else sales[sales[column] == value]
        try:
            series = make_series(data, freq)
            with st.spinner("Backtesting models..."):
                result = run_forecast(series, freq, int(horizon), models or None, level=level)
            title = f"{'Total sales' if column is None else f'{scope}: {value}'} - {int(horizon)}-{FORECAST_UNIT[freq]} forecast"
            st.session_state["forecast"] = {"source": key, "result": result, "series": series, "title": title}
        except ValueError as exc:
            st.session_state.pop("forecast", None)
            st.error(str(exc))
            return

    state = st.session_state.get("forecast")
    if not state or state["source"] != key:
        st.info("Choose a scope, frequency and horizon, then click Run forecast.")
        return
    result, series = state["result"], state["series"]
    best = result.comparison[result.comparison["selected"]].iloc[0]
    unit = FORECAST_UNIT[result.freq]

    m1, m2, m3, m4 = st.columns([1.6, 1, 1, 1])
    m1.metric("Selected model", MODEL_SHORT.get(best["model"], model_label(best["model"])),
              help=model_label(best["model"]))
    m2.metric("Backtest MAPE", f"{best['mape']:.1f}%", help="Average percentage error on past test periods (lower is better)")
    m3.metric("Error vs seasonal naive", f"{best['vs_baseline_pct']:+.0f}%", help="Positive = lower error than the baseline")
    m4.metric("Forecast total", money(result.forecast["yhat"].sum()),
              help=f"Sum of predicted sales for the next {result.horizon} {unit}s")
    for w in result.warnings:
        st.warning(w)
    if best["model"] != "seasonal_naive" and best["beats_baseline"]:
        st.info(f"**{model_label(best['model'])}** had the lowest backtest error (RMSE {best['rmse']:,.0f}), "
                f"{best['vs_baseline_pct']:.0f}% lower than the seasonal-naive baseline. "
                f"The shaded band is the {result.level:.0%} prediction interval.")

    with card():
        show(C.forecast_chart(series, result.forecast, result.level, state["title"]))

    t_fc, t_cmp, t_about = st.tabs(["Forecast table", "Model comparison", "How it works"])
    with t_fc:
        table = result.forecast.round(0).rename_axis("period").reset_index()
        table["period"] = table["period"].dt.strftime("%Y-%m-%d")
        table = table.rename(columns={"yhat": "forecast", "lower": "lower bound", "upper": "upper bound"})
        st.dataframe(table, hide_index=True, column_config={
            "period": st.column_config.Column("Period"),
            "forecast": st.column_config.NumberColumn("Forecast", format="$%.0f"),
            "lower bound": st.column_config.NumberColumn("Lower bound", format="$%.0f"),
            "upper bound": st.column_config.NumberColumn("Upper bound", format="$%.0f"),
        })
        csv_button("Download forecast (CSV)", table, "forecast.csv", "dl_forecast", primary=True)
    with t_cmp:
        cmp_df = result.comparison.copy()
        cmp_df["model"] = cmp_df["model"].map(model_label)
        cmp_df["selected"] = cmp_df["selected"].map({True: "Selected", False: ""})
        cmp_df = cmp_df.rename(columns={
            "model": "Model", "rmse": "RMSE", "mae": "MAE", "mape": "MAPE (%)", "smape": "sMAPE (%)", "bias": "Bias",
            "folds": "Folds", "vs_baseline_pct": "vs baseline (%)", "beats_baseline": "Beats baseline", "selected": "Selected"})
        cmp_df[["RMSE", "MAE", "Bias"]] = cmp_df[["RMSE", "MAE", "Bias"]].round(0)
        cmp_df[["MAPE (%)", "sMAPE (%)", "vs baseline (%)"]] = cmp_df[["MAPE (%)", "sMAPE (%)", "vs baseline (%)"]].round(1)
        st.dataframe(cmp_df, hide_index=True)
        st.caption("Ranked by average RMSE over rolling-origin folds. Models within 1% of the best RMSE are separated by MAPE.")
    with t_about:
        st.markdown(
            "- **Backtesting:** each model is trained on the past only and tested on the following periods, "
            "repeated over several expanding windows (never random splits).\n"
            "- **Baseline:** every model must beat *seasonal naive* (repeat last season).\n"
            "- **Interval:** built from the selected model's backtest errors, so it reflects how wrong the model has been.\n"
            "- **Caveat:** forecasts assume the future resembles the past; promotions, launches or shocks are not modelled."
        )


def page_data(raw: pd.DataFrame, result) -> None:
    page_header("Data and quality report", "What was checked and fixed before analysis, plus the raw and cleaned data")
    r = result.report
    cols = st.columns(5)
    cols[0].metric("Rows in", f"{r.rows_in:,}")
    cols[1].metric("Clean sales rows", f"{r.rows_out:,}")
    cols[2].metric("Duplicates removed", f"{r.duplicates_removed:,}")
    cols[3].metric("Returns separated", f"{r.returns_separated:,}")
    cols[4].metric("Outliers " + ("flagged" if r.outlier_action == "flag" else r.outlier_action), f"{r.outliers_flagged:,}")
    st.caption(f"Date range {r.date_min} to {r.date_max}. Invalid rows dropped: {r.invalid_rows_dropped:,}. "
               f"Unit prices derived: {r.unit_price_derived:,}.")
    with st.expander("What was fixed or filled in"):
        if r.missing_values_filled:
            st.write("Missing values filled with 'Unknown' / defaults:", r.missing_values_filled)
        if r.optional_columns_added:
            st.write("Optional columns not in the file (added empty):", r.optional_columns_added)
        if r.columns_renamed:
            st.write("Columns renamed automatically:", r.columns_renamed)
        if not (r.missing_values_filled or r.optional_columns_added or r.columns_renamed):
            st.write("Nothing needed to be filled or renamed.")

    with card():
        head_left, head_right = st.columns([3, 1])
        head_left.markdown("**Data preview** (first 100 rows)")
        with head_right:
            csv_button("Download cleaned data (CSV)", result.sales, "cleaned_sales.csv", "dl_clean", primary=True)
        tab_clean, tab_raw = st.tabs(["Cleaned sales data", "Raw data"])
        with tab_clean:
            st.dataframe(result.sales.head(100))
        with tab_raw:
            st.dataframe(raw.head(100))


# ------------------------------------------------------------------------ main
def main() -> None:
    st.set_page_config(page_title="Sales Analytics", page_icon="📈", layout="wide")
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
    st.sidebar.title("Sales Analytics")
    st.sidebar.caption("Dashboard and forecasting")
    page = st.sidebar.radio("Page", ["Overview", "Sales analysis", "Forecast Studio", "Data"], key="page",
                            format_func=lambda p: NAV_ICONS[p] + p, label_visibility="collapsed")
    key, raw, result, conn = sidebar_data()
    filters = sidebar_filters(key, conn, result.sales)
    if page == "Overview":
        page_overview(conn, filters)
    elif page == "Sales analysis":
        page_analysis(conn, filters)
    elif page == "Forecast Studio":
        page_forecast(key, result.sales)
    else:
        page_data(raw, result)


if __name__ == "__main__":
    main()