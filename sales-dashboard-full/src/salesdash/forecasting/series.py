"""Turn cleaned transactions into a regular time series."""
from __future__ import annotations

import pandas as pd

SEASON_LENGTH = {"D": 7, "W": 52, "M": 12}
_RULE = {"D": "D", "W": "W-MON", "M": "MS"}


# A first/last period counts as partial only if the data misses MORE than this many days of it.
# (Real files often start on Jan 3 or end on Dec 30 simply because nothing sold on the edge days.)
PARTIAL_TOLERANCE_DAYS = {"W": 2, "M": 3}


def make_series(sales: pd.DataFrame, freq: str = "M", value: str = "sales_amount",
                drop_partial: bool = True, tolerance_days: int | None = None) -> pd.Series:
    """Aggregate transactions to a gap-free series (missing periods become 0).

    freq: "D", "W" (weeks start Monday) or "M" (month start).
    drop_partial: drop the first/last period if the data misses more than `tolerance_days`
    of it (default 3 days for months, 2 for weeks).
    """
    if freq not in _RULE:
        raise ValueError(f"freq must be one of {sorted(_RULE)}")
    s = sales.set_index("order_date")[value].astype(float).sort_index()
    lo, hi = s.index.min(), s.index.max()
    if freq == "D":
        out = s.resample("D").sum()
    elif freq == "W":
        out = s.resample("W-MON", label="left", closed="left").sum()
    else:
        out = s.resample("MS").sum()
    out.index.freq = pd.tseries.frequencies.to_offset(_RULE[freq])

    if drop_partial and len(out) > 2 and freq != "D":
        first_start = out.index[0]
        last_end = out.index[-1] + (pd.offsets.Week(1) if freq == "W" else pd.offsets.MonthBegin(1)) - pd.Timedelta(days=1)
        tol = PARTIAL_TOLERANCE_DAYS[freq] if tolerance_days is None else tolerance_days
        if (lo - first_start).days > tol:
            out = out.iloc[1:]
        if (last_end - hi).days > tol:
            out = out.iloc[:-1]
    out.name = value
    return out
