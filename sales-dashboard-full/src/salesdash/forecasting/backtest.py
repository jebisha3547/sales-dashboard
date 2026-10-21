"""Rolling-origin (expanding-window) backtesting and model comparison.

Never uses random splits: each fold trains only on data before its test window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import metrics
from .models import ALL_MODELS, BASELINE, available_models
from .series import SEASON_LENGTH


@dataclass
class BacktestResult:
    model: str
    folds: pd.DataFrame      # one row per fold with its metrics
    residuals: np.ndarray    # pooled (actual - forecast) across folds, for intervals
    actual: np.ndarray
    predicted: np.ndarray


def fold_starts(n: int, horizon: int, n_folds: int, step: int, min_train: int) -> list[int]:
    """Start index of each test window; reduces folds if history is too short."""
    for k in range(n_folds, 0, -1):
        first = n - horizon - (k - 1) * step
        if first >= min_train:
            return [first + i * step for i in range(k)]
    raise ValueError(
        f"Not enough history: need at least {min_train + horizon} periods for horizon {horizon}, have {n}."
    )


def rolling_origin(series: pd.Series, model_fn, horizon: int, season: int,
                   n_folds: int = 3, step: int | None = None,
                   min_train: int | None = None) -> BacktestResult:
    step = step or horizon
    min_train = min_train or min(2 * season, max(season + 1, len(series) // 2))
    starts = fold_starts(len(series), horizon, n_folds, step, min_train)

    rows, resid, act, pred = [], [], [], []
    for k, s in enumerate(starts, 1):
        train, test = series.iloc[:s], series.iloc[s:s + horizon]
        yhat = np.asarray(model_fn(train, len(test), season), dtype=float)
        rows.append({"fold": k, "train_end": train.index[-1], "test_start": test.index[0], **metrics.all_metrics(test, yhat)})
        resid.append(test.to_numpy(dtype=float) - yhat)
        act.append(test.to_numpy(dtype=float))
        pred.append(yhat)
    return BacktestResult(getattr(model_fn, "__name__", "model"), pd.DataFrame(rows),
                          np.concatenate(resid), np.concatenate(act), np.concatenate(pred))


def compare_models(series: pd.Series, freq: str, horizon: int, models: list[str] | None = None,
                   n_folds: int = 3, tie_tolerance: float = 0.01):
    """Backtest every model and rank them.

    Selection rule: lowest average RMSE; among models within `tie_tolerance` (1%) of the
    best RMSE, the one with the lowest MAPE wins.

    Returns (comparison_table, {model: BacktestResult}, warnings).
    """
    season = SEASON_LENGTH[freq]
    names = models or available_models()
    if BASELINE not in names:
        names = [BASELINE] + list(names)

    results, warnings = {}, []
    for name in names:
        try:
            results[name] = rolling_origin(series, ALL_MODELS[name], horizon, season, n_folds)
            results[name].model = name
        except ValueError:
            raise
        except Exception as exc:  # a failing model must not break the whole comparison
            warnings.append(f"Model '{name}' skipped: {exc}")

    rows = []
    for name, res in results.items():
        f = res.folds
        rows.append({"model": name, "rmse": f["rmse"].mean(), "mae": f["mae"].mean(),
                     "mape": f["mape"].mean(), "smape": f["smape"].mean(),
                     "bias": f["bias"].mean(), "folds": len(f)})
    table = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)

    base_rmse = table.loc[table["model"] == BASELINE, "rmse"].iloc[0]
    table["vs_baseline_pct"] = (1 - table["rmse"] / base_rmse) * 100   # + = better than seasonal naive
    table["beats_baseline"] = table["rmse"] < base_rmse

    best_rmse = table["rmse"].min()
    contenders = table[table["rmse"] <= best_rmse * (1 + tie_tolerance)]
    best = contenders.sort_values("mape").iloc[0]["model"]
    table["selected"] = table["model"] == best

    if not (table["beats_baseline"].any()):
        warnings.append("No model beat the seasonal-naive baseline: the data may be too short or too noisy to forecast reliably.")
    return table, results, warnings
