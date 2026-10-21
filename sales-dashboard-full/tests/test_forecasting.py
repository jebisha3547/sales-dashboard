import numpy as np
import pandas as pd
import pytest

from helpers import raw_sales, seasonal_series
from salesdash.cleaning import clean_sales_data
from salesdash.forecasting import compare_models, make_series, rolling_origin, run_forecast
from salesdash.forecasting import metrics
from salesdash.forecasting.backtest import fold_starts
from salesdash.forecasting.forecast import interval_offsets
from salesdash.forecasting.models import available_models, naive, seasonal_naive, trend_seasonal_reg


def test_metrics_known_values():
    y, yhat = [100, 200, 300], [110, 190, 330]
    assert metrics.mae(y, yhat) == pytest.approx(50 / 3)
    assert metrics.rmse(y, yhat) == pytest.approx(np.sqrt((100 + 100 + 900) / 3))
    assert metrics.mape(y, yhat) == pytest.approx((10 + 5 + 10) / 3)
    assert metrics.bias(y, yhat) == pytest.approx(10.0)   # errors: +10, -10, +30
    assert metrics.coverage([1, 5, 9], [0, 0, 0], [2, 6, 8]) == pytest.approx(200 / 3)


def test_mape_ignores_zero_actuals():
    assert metrics.mape([0, 100], [5, 110]) == pytest.approx(10.0)


def test_naive_and_seasonal_naive():
    s = pd.Series(range(1, 25), index=pd.date_range("2020-01-01", periods=24, freq="MS"), dtype=float)
    assert list(naive(s, 3, 12)) == [24, 24, 24]
    assert list(seasonal_naive(s, 14, 12)) == list(range(13, 25)) + [13, 14]


def test_trend_seasonal_reg_recovers_a_clean_pattern():
    s = seasonal_series(n=48, noise=0.0)
    full = seasonal_series(n=60, noise=0.0)
    pred = trend_seasonal_reg(s, 12, 12)
    assert metrics.mape(full.iloc[48:], pred) < 5


def test_make_series_monthly_drops_partial_months():
    df = clean_sales_data(raw_sales(n_days=100, start="2023-01-15")).sales   # starts mid-Jan, ends 25 Apr
    s = make_series(df, "M")
    assert s.index[0] == pd.Timestamp("2023-02-01")
    assert s.index[-1] == pd.Timestamp("2023-03-01")
    kept = make_series(df, "M", drop_partial=False)
    assert kept.index[0] == pd.Timestamp("2023-01-01")


def test_make_series_keeps_period_when_only_edge_days_are_missing():
    # real files often start on Jan 3 and end on Dec 30: those months must be kept
    df = clean_sales_data(raw_sales(n_days=363, start="2023-01-03")).sales          # Jan 3 - Dec 31 2023
    df = df[df["order_date"] <= "2023-12-30"]
    s = make_series(df, "M")
    assert s.index[0] == pd.Timestamp("2023-01-01")
    assert s.index[-1] == pd.Timestamp("2023-12-01")
    assert len(s) == 12


def test_make_series_fills_gaps_with_zero_and_is_regular():
    df = clean_sales_data(raw_sales(n_days=60)).sales
    df = df[df["order_date"].dt.day != 10]
    s = make_series(df, "D")
    assert len(s) == 60
    assert s.index.freq is not None
    assert (s > 0).all() or (s == 0).any()


def test_make_series_total_matches_when_nothing_dropped():
    df = clean_sales_data(raw_sales(n_days=90, start="2023-01-01")).sales   # Jan 1 - Mar 31 exactly
    s = make_series(df, "M")
    assert s.sum() == pytest.approx(df["sales_amount"].sum())


def test_fold_starts_reduces_folds_and_errors_when_too_short():
    assert fold_starts(48, 6, 3, 6, 24) == [30, 36, 42]
    assert fold_starts(48, 12, 3, 12, 24) == [24, 36]
    with pytest.raises(ValueError):
        fold_starts(20, 12, 3, 12, 24)


def test_backtest_never_leaks_future_data():
    s = seasonal_series(n=48)
    seen = []

    def spy(train, horizon, season):
        seen.append((train.index[-1], len(train)))
        assert train.index.max() < s.index[len(train)]       # train strictly precedes its test window
        return np.repeat(train.iloc[-1], horizon)

    res = rolling_origin(s, spy, horizon=6, season=12, n_folds=3)
    assert [n for _, n in seen] == [30, 36, 42]               # expanding window
    assert len(res.folds) == 3 and len(res.residuals) == 18


def test_compare_models_ranks_and_selects_best_over_baseline():
    s = seasonal_series(n=60, trend=3.0, noise=2.0)
    table, results, warns = compare_models(s, "M", horizon=6)
    assert table.iloc[0]["rmse"] == table["rmse"].min()
    assert table["selected"].sum() == 1
    assert "seasonal_naive" in set(table["model"])
    assert bool(table.loc[table["selected"], "beats_baseline"].iloc[0])
    assert warns == []


def test_selection_uses_mape_to_break_rmse_ties():
    s = seasonal_series(n=60)
    table, _, _ = compare_models(s, "M", horizon=6, tie_tolerance=10.0)   # everything is a "tie"
    chosen = table.loc[table["selected"], "model"].iloc[0]
    assert chosen == table.sort_values("mape").iloc[0]["model"]


def test_warns_when_nothing_beats_the_baseline():
    s = seasonal_series(n=60)
    table, _, warns = compare_models(s, "M", horizon=6, models=["seasonal_naive"])
    assert any("baseline" in w for w in warns)


def test_run_forecast_shapes_dates_and_interval_order():  # TC-04 (12-month horizon)
    s = seasonal_series(n=60)
    res = run_forecast(s, "M", horizon=12)
    fc = res.forecast
    assert len(fc) == 12
    assert fc.index[0] == s.index[-1] + pd.offsets.MonthBegin(1)
    assert (fc["lower"] <= fc["upper"]).all()
    assert (fc["lower"] >= 0).all()
    assert (fc["yhat"] >= fc["lower"]).all()


def test_interval_widens_with_confidence_level():
    resid = np.random.default_rng(0).normal(0, 10, 100)
    lo80, hi80 = interval_offsets(resid, 0.80)
    lo95, hi95 = interval_offsets(resid, 0.95)
    assert lo95 < lo80 < hi80 < hi95


def test_interval_normal_fallback_with_few_residuals():
    lo, hi = interval_offsets(np.array([-5.0, 5.0, -3.0, 3.0]), 0.95)
    assert lo < 0 < hi


def test_too_little_history_gives_clear_error():
    s = seasonal_series(n=20)
    with pytest.raises(ValueError, match="Not enough history"):
        run_forecast(s, "M", horizon=12)


def test_optional_models_only_listed_when_installed():
    names = available_models()
    assert {"naive", "seasonal_naive", "trend_seasonal_reg", "gbm_trend_lags"} <= set(names)
