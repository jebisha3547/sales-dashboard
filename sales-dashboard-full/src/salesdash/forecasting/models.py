"""Forecast models. Each is a function (train: pd.Series, horizon: int, season: int) -> np.ndarray.

Always available (numpy + scikit-learn):
    naive, seasonal_naive, trend_seasonal_reg, gbm_trend_lags
Optional (used automatically if the library is installed):
    holt_winters (statsmodels), prophet (prophet)
"""
from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

BASELINE = "seasonal_naive"


class ModelUnavailable(ImportError):
    pass


def naive(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    return np.repeat(float(train.iloc[-1]), horizon)


def seasonal_naive(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    """Repeat the last full season. Every other model has to beat this."""
    if len(train) < season:
        return naive(train, horizon, season)
    last = train.to_numpy(dtype=float)[-season:]
    return np.array([last[i % season] for i in range(horizon)])


def _design(start: int, n: int, season: int) -> np.ndarray:
    t = np.arange(start, start + n)
    dummies = [(t % season == k).astype(float) for k in range(season)]
    return np.column_stack([t.astype(float)] + dummies)


def trend_seasonal_reg(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    """Log-linear trend + seasonal dummies (multiplicative seasonality, extrapolates the trend)."""
    n = len(train)
    if n < 2 * season:
        return seasonal_naive(train, horizon, season)
    y = np.log1p(train.to_numpy(dtype=float))
    model = LinearRegression(fit_intercept=False).fit(_design(0, n, season), y)
    return np.maximum(np.expm1(model.predict(_design(n, horizon, season))), 0.0)


def _gbm_features(hist: list[float], i: int, lags: list[int], season: int) -> list[float]:
    feats = [hist[i - lag] for lag in lags]
    feats.append(float(np.mean(hist[i - 3:i])))
    feats += [float(i % season), np.sin(2 * np.pi * i / season), np.cos(2 * np.pi * i / season)]
    return feats


def gbm_trend_lags(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    """Gradient boosting on lag / rolling / calendar features, on top of a linear trend.

    Trees cannot extrapolate a trend, so a log-linear trend is removed first and the
    boosted model learns the remaining (seasonal) structure. Forecasts are recursive.
    """
    y = np.log1p(train.to_numpy(dtype=float))
    n = len(y)
    lags = [1, 2, 3] + ([season] if n >= 2 * season else [])
    max_lag = max(lags)
    if n - max_lag < 12:
        return seasonal_naive(train, horizon, season)

    t = np.arange(n)
    coef = np.polyfit(t, y, 1)
    resid = list(y - np.polyval(coef, t))
    X = np.array([_gbm_features(resid, i, lags, season) for i in range(max_lag, n)])
    target = np.array(resid[max_lag:])
    model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                      subsample=1.0, random_state=0).fit(X, target)

    hist, preds = list(resid), []
    for h in range(horizon):
        i = n + h
        r = float(model.predict(np.array([_gbm_features(hist, i, lags, season)]))[0])
        hist.append(r)
        preds.append(np.expm1(np.polyval(coef, i) + r))
    return np.maximum(np.array(preds), 0.0)


def holt_winters(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    """Damped-trend Holt-Winters with multiplicative seasonality (needs statsmodels)."""
    if importlib.util.find_spec("statsmodels") is None:
        raise ModelUnavailable("statsmodels is not installed")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    if len(train) < 2 * season:
        return seasonal_naive(train, horizon, season)
    y = train.astype(float).clip(lower=1e-6).reset_index(drop=True)
    fit = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="mul",
                               seasonal_periods=season, initialization_method="estimated").fit()
    return np.maximum(np.asarray(fit.forecast(horizon)), 0.0)


def prophet(train: pd.Series, horizon: int, season: int) -> np.ndarray:
    """Facebook Prophet (needs the prophet package)."""
    if importlib.util.find_spec("prophet") is None:
        raise ModelUnavailable("prophet is not installed")
    from prophet import Prophet

    freq = train.index.freq or pd.infer_freq(train.index)
    df = pd.DataFrame({"ds": train.index, "y": train.to_numpy(dtype=float)})
    m = Prophet(weekly_seasonality=(season == 7), yearly_seasonality=True, daily_seasonality=False)
    m.fit(df)
    future = pd.DataFrame({"ds": pd.date_range(train.index[-1], periods=horizon + 1, freq=freq)[1:]})
    return np.maximum(m.predict(future)["yhat"].to_numpy(), 0.0)


ALL_MODELS: dict[str, Callable] = {
    "naive": naive,
    "seasonal_naive": seasonal_naive,
    "trend_seasonal_reg": trend_seasonal_reg,
    "gbm_trend_lags": gbm_trend_lags,
    "holt_winters": holt_winters,
    "prophet": prophet,
}
_REQUIRES = {"holt_winters": "statsmodels", "prophet": "prophet"}


def available_models() -> list[str]:
    """Models whose dependencies are installed."""
    return [name for name in ALL_MODELS
            if name not in _REQUIRES or importlib.util.find_spec(_REQUIRES[name]) is not None]
