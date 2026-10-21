"""End-to-end forecast: backtest, select, forecast with prediction intervals."""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

from .backtest import BacktestResult, compare_models
from .models import ALL_MODELS
from .series import SEASON_LENGTH


@dataclass
class ForecastResult:
    best_model: str
    comparison: pd.DataFrame            # all models, ranked
    forecast: pd.DataFrame              # index = future periods; yhat, lower, upper
    level: float
    horizon: int
    freq: str
    warnings: list = field(default_factory=list)
    backtests: dict = field(default_factory=dict)


def interval_offsets(residuals: np.ndarray, level: float) -> tuple[float, float]:
    """Lower/upper offsets from backtest residuals (actual - forecast).

    Uses empirical quantiles when there are enough residuals, otherwise a normal approximation.
    """
    alpha = (1 - level) / 2
    if len(residuals) >= 20:
        return float(np.quantile(residuals, alpha)), float(np.quantile(residuals, 1 - alpha))
    z = NormalDist().inv_cdf(1 - alpha)
    mu, sd = float(np.mean(residuals)), float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0
    return mu - z * sd, mu + z * sd


def run_forecast(series: pd.Series, freq: str = "M", horizon: int = 6, models: list[str] | None = None,
                 n_folds: int = 3, level: float = 0.80) -> ForecastResult:
    """Backtest candidate models, pick the best, and forecast `horizon` periods ahead."""
    if freq not in SEASON_LENGTH:
        raise ValueError(f"freq must be one of {sorted(SEASON_LENGTH)}")
    season = SEASON_LENGTH[freq]
    table, results, warnings = compare_models(series, freq, horizon, models, n_folds)
    best = str(table.loc[table["selected"], "model"].iloc[0])

    yhat = np.asarray(ALL_MODELS[best](series, horizon, season), dtype=float)
    lo_off, hi_off = interval_offsets(results[best].residuals, level)
    future = pd.date_range(series.index[-1], periods=horizon + 1, freq=series.index.freq)[1:]
    fc = pd.DataFrame({"yhat": yhat,
                       "lower": np.maximum(yhat + lo_off, 0.0),
                       "upper": yhat + hi_off}, index=future)
    return ForecastResult(best, table, fc, level, horizon, freq, warnings, results)
