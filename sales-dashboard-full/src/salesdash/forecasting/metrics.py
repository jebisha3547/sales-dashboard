"""Forecast error metrics."""
from __future__ import annotations

import numpy as np


def _arr(x):
    return np.asarray(x, dtype=float)


def mae(y, yhat) -> float:
    return float(np.mean(np.abs(_arr(y) - _arr(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((_arr(y) - _arr(yhat)) ** 2)))


def mape(y, yhat) -> float:
    """Mean absolute percentage error (%), ignoring periods where actual is 0."""
    y, yhat = _arr(y), _arr(yhat)
    m = y != 0
    return float(np.mean(np.abs((y[m] - yhat[m]) / y[m])) * 100) if m.any() else float("nan")


def smape(y, yhat) -> float:
    y, yhat = _arr(y), _arr(yhat)
    denom = (np.abs(y) + np.abs(yhat)) / 2
    m = denom != 0
    return float(np.mean(np.abs(y[m] - yhat[m]) / denom[m]) * 100) if m.any() else float("nan")


def bias(y, yhat) -> float:
    """Mean error (forecast - actual). Positive = over-forecasting."""
    return float(np.mean(_arr(yhat) - _arr(y)))


def coverage(y, lower, upper) -> float:
    """Share of actuals inside the prediction interval (%)."""
    y, lo, hi = _arr(y), _arr(lower), _arr(upper)
    return float(np.mean((y >= lo) & (y <= hi)) * 100)


def all_metrics(y, yhat) -> dict:
    return {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "mape": mape(y, yhat),
            "smape": smape(y, yhat), "bias": bias(y, yhat)}
