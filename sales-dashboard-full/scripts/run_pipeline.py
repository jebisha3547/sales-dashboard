"""One-command demo: clean -> store -> KPIs -> forecast comparison.

    python scripts/run_pipeline.py --csv data/superstore_synthetic.csv --freq M --horizon 6
"""
import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from salesdash import analytics as A  # noqa: E402
from salesdash import db  # noqa: E402
from salesdash.cleaning import load_and_clean_csv  # noqa: E402
from salesdash.forecasting import make_series, run_forecast  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(ROOT / "data" / "superstore_synthetic.csv"))
    p.add_argument("--freq", choices=["D", "W", "M"], default="M")
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--level", type=float, default=0.80)
    p.add_argument("--dayfirst", action="store_true", help="dates are day-first (dd/mm/yyyy)")
    a = p.parse_args()
    pd.set_option("display.width", 200)

    print("== 1. Cleaning report ==")
    res = load_and_clean_csv(a.csv, dayfirst=a.dayfirst)
    print(res.report.summary())

    conn = db.connect()
    db.load_transactions(conn, res.sales, res.returns)

    print("\n== 2. KPIs ==")
    k = A.kpis(conn)
    for name, v in k["current"].items():
        g = (k["growth_pct"] or {}).get(name)
        print(f"{name:>16}: {v:>14,.2f}" + (f"   ({g:+.1f}% vs previous period)" if g is not None else ""))
    print(f"{'return_rate':>16}: {k['return_rate']:.2%}")
    print("\nRevenue by category:\n", A.breakdown(conn, "category")[["category", "revenue", "profit", "share_pct"]].round(1).to_string(index=False))

    print(f"\n== 3. Forecast ({a.freq}, horizon {a.horizon}) ==")
    series = make_series(res.sales, a.freq)
    out = run_forecast(series, a.freq, a.horizon, level=a.level)
    cols = ["model", "rmse", "mae", "mape", "bias", "vs_baseline_pct", "beats_baseline", "selected"]
    print(out.comparison[cols].round(2).to_string(index=False))
    print(f"\nSelected model: {out.best_model}")
    for w in out.warnings:
        print("WARNING:", w)
    print(f"\nForecast with {out.level:.0%} prediction interval:")
    print(out.forecast.round(0).to_string())

    outdir = ROOT / "outputs"
    outdir.mkdir(exist_ok=True)
    out.comparison.to_csv(outdir / f"model_comparison_{a.freq}.csv", index=False)
    out.forecast.to_csv(outdir / f"forecast_{a.freq}.csv")
    print(f"\nSaved results to {outdir}/")


if __name__ == "__main__":
    main()
