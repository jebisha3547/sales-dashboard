"""Small builders shared by the tests."""
import numpy as np
import pandas as pd


def raw_sales(n_days: int = 60, per_day: int = 3, start: str = "2023-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows, oid = [], 1
    for d in pd.date_range(start, periods=n_days):
        for _ in range(per_day):
            qty = int(rng.integers(1, 5))
            price = float(rng.choice([10.0, 25.0, 100.0]))
            cat = str(rng.choice(["Furniture", "Technology", "Office Supplies"]))
            rows.append({
                "order_id": f"O{oid}", "order_date": d.strftime("%Y-%m-%d"),
                "product_id": f"P{rng.integers(1, 6)}", "product_name": "Widget",
                "category": cat, "quantity": qty, "unit_price": price,
                "sales_amount": qty * price, "profit": qty * price * 0.2,
                "region": str(rng.choice(["West", "East"])), "customer_id": f"C{rng.integers(1, 10)}",
                "sales_channel": "Online",
            })
            oid += 1
    return pd.DataFrame(rows)


def seasonal_series(n: int = 60, season: int = 12, trend: float = 2.0, noise: float = 3.0, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 200 + trend * t + 40 * np.sin(2 * np.pi * t / season) + rng.normal(0, noise, n)
    return pd.Series(y, index=pd.date_range("2020-01-01", periods=n, freq="MS"))
