"""SQLite storage and pre-aggregated summary tables.

SQLite keeps the project zero-setup. The schema and SQL are standard, so moving to
PostgreSQL later means swapping the connection, not rewriting the queries.
"""
from __future__ import annotations

import sqlite3

import pandas as pd


def connect(path: str = ":memory:") -> sqlite3.Connection:
    return sqlite3.connect(path, check_same_thread=False)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].astype(object)
    return out


def load_transactions(conn: sqlite3.Connection, sales: pd.DataFrame, returns: pd.DataFrame | None = None) -> None:
    """Write cleaned sales (and returns) to the database, add indexes, build aggregates."""
    _prepare(sales).to_sql("sales_transactions", conn, if_exists="replace", index=False)
    ret = returns if returns is not None else sales.head(0)
    _prepare(ret).to_sql("sales_returns", conn, if_exists="replace", index=False)
    for table in ("sales_transactions", "sales_returns"):
        for col in ("order_date", "category", "region", "sales_channel", "customer_id"):
            conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}({col})")
    conn.commit()
    build_aggregates(conn)


_AGG_SELECT = (
    "SUM(sales_amount) AS revenue, SUM(profit) AS profit, SUM(quantity) AS units, "
    "COUNT(DISTINCT order_id) AS orders"
)
_PERIOD_EXPR = {
    "sales_daily": "order_date",
    "sales_weekly": "date(order_date, 'weekday 0', '-6 days')",  # Monday of the week
    "sales_monthly": "strftime('%Y-%m-01', order_date)",
}


def build_aggregates(conn: sqlite3.Connection) -> None:
    """Create sales_daily / sales_weekly / sales_monthly summary tables."""
    for table, expr in _PERIOD_EXPR.items():
        conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            f"CREATE TABLE {table} AS SELECT {expr} AS period, {_AGG_SELECT} "
            f"FROM sales_transactions GROUP BY 1 ORDER BY 1"
        )
    conn.commit()
