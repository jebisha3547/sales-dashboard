"""KPI, trend, breakdown and customer analytics (SQL over the transactions table).

Every user-controlled value goes through query parameters; column names and period
expressions come from fixed whitelists, so filters cannot inject SQL.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd

PERIOD_SQL = {
    "D": "order_date",
    "W": "date(order_date, 'weekday 0', '-6 days')",
    "M": "strftime('%Y-%m-01', order_date)",
    "Y": "strftime('%Y-01-01', order_date)",
}
DIMENSIONS = {
    "category", "sub_category", "region", "state", "city", "customer_segment",
    "sales_channel", "product_name", "product_id", "shipping_mode", "country",
}
METRICS = {
    "revenue": "SUM(sales_amount)",
    "profit": "SUM(profit)",
    "units": "SUM(quantity)",
    "orders": "COUNT(DISTINCT order_id)",
}
_KPI_SELECT = (
    "COALESCE(SUM(sales_amount),0) AS revenue, COALESCE(SUM(profit),0) AS profit, "
    "COALESCE(SUM(quantity),0) AS units, COUNT(DISTINCT order_id) AS orders"
)


@dataclass(frozen=True)
class Filters:
    """Global dashboard filters. Dates are ISO strings (inclusive)."""
    start: str | None = None
    end: str | None = None
    categories: tuple = ()
    regions: tuple = ()
    channels: tuple = ()
    segments: tuple = ()


def _where(f: Filters) -> tuple[str, list]:
    clauses, params = [], []
    if f.start:
        clauses.append("order_date >= ?"); params.append(f.start)
    if f.end:
        clauses.append("order_date <= ?"); params.append(f.end)
    for col, vals in (("category", f.categories), ("region", f.regions),
                      ("sales_channel", f.channels), ("customer_segment", f.segments)):
        if vals:
            clauses.append(f"{col} IN ({','.join('?' * len(vals))})")
            params.extend(vals)
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def _query(conn: sqlite3.Connection, sql: str, params: list | None = None) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params or [])


def date_bounds(conn: sqlite3.Connection) -> tuple[str, str]:
    row = conn.execute("SELECT MIN(order_date), MAX(order_date) FROM sales_transactions").fetchone()
    return row[0], row[1]


def _totals(conn, f: Filters) -> dict:
    where, params = _where(f)
    row = _query(conn, f"SELECT {_KPI_SELECT} FROM sales_transactions{where}", params).iloc[0]
    rev, orders = float(row["revenue"]), int(row["orders"])
    return {
        "revenue": rev,
        "profit": float(row["profit"]),
        "units": int(row["units"]),
        "orders": orders,
        "avg_order_value": rev / orders if orders else 0.0,
        "profit_margin": float(row["profit"]) / rev if rev else 0.0,
    }


def _growth(cur: dict, prev: dict) -> dict:
    return {k: ((cur[k] - prev[k]) / prev[k] * 100 if prev[k] else None) for k in cur}


def kpis(conn: sqlite3.Connection, f: Filters = Filters()) -> dict:
    """Headline KPIs for the filtered window plus growth vs the previous equal-length window."""
    lo, hi = date_bounds(conn)
    start, end = f.start or lo, f.end or hi
    current = _totals(conn, replace(f, start=start, end=end))

    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    length = (d1 - d0).days + 1
    prev_end = d0 - timedelta(days=1)
    prev_start = prev_end - timedelta(days=length - 1)
    previous = growth = None
    if prev_end.isoformat() >= lo:  # only compare if prior data exists
        previous = _totals(conn, replace(f, start=prev_start.isoformat(), end=prev_end.isoformat()))
        growth = _growth(current, previous)

    where, params = _where(replace(f, start=start, end=end))
    ret = _query(conn, f"SELECT COALESCE(SUM(ABS(quantity)),0) AS u FROM sales_returns{where}", params).iloc[0]["u"]
    return_rate = float(ret) / current["units"] if current["units"] else 0.0

    return {
        "period": (start, end),
        "current": current,
        "previous": previous,
        "previous_period": (prev_start.isoformat(), prev_end.isoformat()) if previous else None,
        "growth_pct": growth,
        "return_rate": return_rate,
    }


def trends(conn: sqlite3.Connection, freq: str = "M", f: Filters = Filters()) -> pd.DataFrame:
    """Revenue/profit/units/orders per period. freq: D, W (Mon-start), M or Y."""
    if freq not in PERIOD_SQL:
        raise ValueError(f"freq must be one of {sorted(PERIOD_SQL)}")
    where, params = _where(f)
    sql = (f"SELECT {PERIOD_SQL[freq]} AS period, {_KPI_SELECT} FROM sales_transactions"
           f"{where} GROUP BY 1 ORDER BY 1")
    df = _query(conn, sql, params)
    df["period"] = pd.to_datetime(df["period"])
    return df


def breakdown(conn, dimension: str, f: Filters = Filters(), metric: str = "revenue",
              top_n: int | None = None, ascending: bool = False) -> pd.DataFrame:
    """Metric by a dimension (category, region, product_name, ...) with share of total."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {sorted(DIMENSIONS)}")
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {sorted(METRICS)}")
    where, params = _where(f)
    sql = (f"SELECT {dimension} AS {dimension}, {_KPI_SELECT} FROM sales_transactions"
           f"{where} GROUP BY 1 ORDER BY {metric} {'ASC' if ascending else 'DESC'}")
    if top_n:
        sql += f" LIMIT {int(top_n)}"
    df = _query(conn, sql, params)
    total = _totals(conn, f)[metric]
    df["share_pct"] = df[metric] / total * 100 if total else 0.0
    return df


def abc_analysis(conn, f: Filters = Filters(), dimension: str = "product_name",
                 a_cut: float = 80.0, b_cut: float = 95.0) -> pd.DataFrame:
    """ABC classes by cumulative revenue share: A <= 80%, B <= 95%, C the rest."""
    df = breakdown(conn, dimension, f).reset_index(drop=True)
    df["cum_share_pct"] = df["share_pct"].cumsum()
    # an item is class A if the cumulative share *before* it is still below the cut
    before = df["cum_share_pct"] - df["share_pct"]
    df["abc_class"] = "C"
    df.loc[before < b_cut, "abc_class"] = "B"
    df.loc[before < a_cut, "abc_class"] = "A"
    return df


def new_vs_returning(conn, f: Filters = Filters()) -> pd.DataFrame:
    """Monthly count of customers placing their first order (new) vs repeat customers."""
    where, params = _where(f)
    sql = f"""
        WITH first AS (
            SELECT customer_id, strftime('%Y-%m', MIN(order_date)) AS first_month
            FROM sales_transactions GROUP BY customer_id)
        SELECT strftime('%Y-%m-01', s.order_date) AS period,
               COUNT(DISTINCT CASE WHEN strftime('%Y-%m', s.order_date) = f.first_month
                                   THEN s.customer_id END) AS new_customers,
               COUNT(DISTINCT CASE WHEN strftime('%Y-%m', s.order_date) <> f.first_month
                                   THEN s.customer_id END) AS returning_customers
        FROM sales_transactions s JOIN first f ON s.customer_id = f.customer_id
        {where} GROUP BY 1 ORDER BY 1"""
    df = _query(conn, sql, params)
    df["period"] = pd.to_datetime(df["period"])
    return df


def anomalies(conn, freq: str = "D", f: Filters = Filters(), z: float = 3.0) -> pd.DataFrame:
    """Flag periods whose revenue deviates from a centred rolling median by > z robust sigmas."""
    df = trends(conn, freq, f)
    if len(df) < 8:
        df["is_anomaly"] = False
        return df
    window = {"D": 28, "W": 13, "M": 6, "Y": 3}[freq]
    med = df["revenue"].rolling(window, center=True, min_periods=3).median()
    resid = df["revenue"] - med
    mad = resid.abs().rolling(window, center=True, min_periods=3).median().replace(0, pd.NA)
    df["robust_z"] = (resid / (1.4826 * mad)).astype(float)
    df["is_anomaly"] = df["robust_z"].abs() > z
    return df
