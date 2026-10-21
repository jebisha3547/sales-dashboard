"""Independent check: recompute key numbers with plain pandas (no salesdash code)."""
import numpy as np
import pandas as pd

df = pd.read_csv("data/superstore_synthetic.csv", encoding="latin-1")   # change path for your own file
df = df.drop_duplicates()
df["order_date"] = pd.to_datetime(df["order_date"])
sales = df[(df["quantity"] > 0) & (df["sales_amount"] > 0)]            # returns excluded

print("--- Overview cards (All time) ---")
rev = sales["sales_amount"].sum(); orders = sales["order_id"].nunique()
print(f"Revenue        {rev:,.2f}")
print(f"Profit         {sales['profit'].sum():,.2f}")
print(f"Orders         {orders:,}")
print(f"Avg order val  {rev / orders:,.2f}")
print(f"Units          {sales['quantity'].sum():,}")
print(f"Profit margin  {sales['profit'].sum() / rev:.1%}")
ret = df[df["quantity"] < 0]
print(f"Return rate    {ret['quantity'].abs().sum() / sales['quantity'].sum():.2%}")

print("\n--- Category share of revenue ---")
print((sales.groupby("category")["sales_amount"].sum() / rev * 100).round(1).sort_values(ascending=False))

print("\n--- Growth: 2024 vs 2023 revenue ---")
by_year = sales.groupby(sales["order_date"].dt.year)["sales_amount"].sum()
print(by_year.round(0).to_dict(), f"-> growth {by_year[2024] / by_year[2023] - 1:+.1%}")

print("\n--- Monthly revenue, last 3 full months ---")
monthly = sales.set_index("order_date")["sales_amount"].resample("MS").sum()
print(monthly.tail(3).round(0))

print("\n--- Seasonal-naive backtest (monthly, horizon 6, 3 folds) ---")
rmses = []
for start in (30, 36, 42):                      # test windows over a 48-month series
    train, test = monthly.iloc[:start], monthly.iloc[start:start + 6]
    pred = train.iloc[-12:].to_numpy()[:6]      # repeat last year's same months
    rmses.append(np.sqrt(np.mean((test.to_numpy() - pred) ** 2)))
print(f"Average RMSE {np.mean(rmses):,.2f}")
