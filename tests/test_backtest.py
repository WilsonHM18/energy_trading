"""Tests for the Phase 5 backtest engine.

All tests use synthetic DataFrames no disk I/O, no network calls.

Helper ``_make_oos_df`` builds a minimal OOS DataFrame:
- ``spread`` constant at 10.0 (so all long-spread trades profit)
- ``spread_pred_*`` varies linearly from -10 to +10 over n_hours
- 4 hubs (default), non-unique UTC DatetimeIndex
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from energy_trading.backtest import (
    compute_hourly_pnl,
    compute_pnl_metrics,
    generate_signals,
    run_all_backtests,
    run_backtest,
)

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_HUBS = ["HB_HOUSTON", "HB_NORTH", "HB_SOUTH", "HB_WEST"]
_MODELS = ["linear", "ridge", "lasso", "lgbm"]

_REQUIRED_TRADE_COLS = {
    "signal",
    "position_mwh",
    "spread_pred",
    "spread_actual",
    "gross_pnl",
    "transaction_cost_paid",
    "net_pnl",
    "cumulative_pnl",
}

_REQUIRED_METRICS_KEYS = {
    "model",
    "hub",
    "threshold",
    "lot_mwh",
    "n_obs",
    "trade_rate",
    "n_trades",
    "total_pnl",
    "annualized_pnl",
    "sharpe",
    "sortino",
    "max_drawdown",
    "win_rate",
    "cvar_5pct",
}


def _make_oos_df(
    n_hours: int = 500,
    hubs: list[str] | None = None,
    spread: float = 10.0,
) -> pd.DataFrame:
    """Synthetic OOS DataFrame with deterministic spread and predictions.

    spread_pred_* varies linearly from -10 to +10 across n_hours.
    The first half of hours has negative predictions, the second half positive.
    spread is constant (default 10.0) so all long-spread trades are profitable.
    """
    hubs = hubs or _HUBS
    idx = pd.date_range("2021-01-01", periods=n_hours, freq="h", tz="UTC")

    preds = np.linspace(-10.0, 10.0, n_hours)

    records = []
    timestamps = []
    for i, ts in enumerate(idx):
        for hub in hubs:
            records.append(
                {
                    "hub": hub,
                    "spread": spread,
                    "spread_pred_linear": preds[i],
                    "spread_pred_ridge": preds[i],
                    "spread_pred_lasso": preds[i],
                    "spread_pred_lgbm": preds[i],
                }
            )
            timestamps.append(ts)

    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex(timestamps, tz="UTC", name="interval_start_utc")
    return df


def _make_single_hub_oos_df(
    n_hours: int = 200,
    pred_value: float = 5.0,
    spread_value: float = 3.0,
) -> pd.DataFrame:
    """Single-hub OOS with constant pred and spread for arithmetic assertions."""
    idx = pd.date_range("2021-01-01", periods=n_hours, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "hub": "HB_NORTH",
            "spread": spread_value,
            "spread_pred_linear": pred_value,
            "spread_pred_ridge": pred_value,
            "spread_pred_lasso": pred_value,
            "spread_pred_lgbm": pred_value,
        },
        index=pd.DatetimeIndex(idx, name="interval_start_utc"),
    )
    return df


# ---------------------------------------------------------------------------
# TestGenerateSignals (8 tests)
# ---------------------------------------------------------------------------


class TestGenerateSignals:
    def test_positive_pred_gives_plus_one(self):
        df = _make_single_hub_oos_df(pred_value=5.0)
        sig = generate_signals(df, model="linear", threshold=0.0)
        assert (sig == 1).all()

    def test_negative_pred_gives_minus_one(self):
        df = _make_single_hub_oos_df(pred_value=-5.0)
        sig = generate_signals(df, model="linear", threshold=0.0)
        assert (sig == -1).all()

    def test_threshold_zero_trades_most_hours(self):
        # With linearly varying pred from -10..+10, only exact-zero pred is flat.
        df = _make_oos_df(n_hours=200, hubs=["HB_NORTH"])
        sig = generate_signals(df, model="linear", threshold=0.0)
        # Almost all hours should be traded (except any where pred exactly == 0)
        n_traded = int((sig != 0).sum())
        assert n_traded >= len(sig) - 1

    def test_threshold_filters_band(self):
        # pred in [-10, 10]; threshold=15 → all flat
        df = _make_oos_df(n_hours=100, hubs=["HB_NORTH"])
        sig = generate_signals(df, model="linear", threshold=15.0)
        assert (sig == 0).all()

    def test_threshold_splits_into_short_long_flat(self):
        # pred linearly -10..+10 over 100 hours; threshold=5
        # First ~quarter: pred < -5 → -1; last ~quarter: pred > +5 → +1; middle: 0
        df = _make_oos_df(n_hours=100, hubs=["HB_NORTH"])
        sig = generate_signals(df, model="linear", threshold=5.0)
        assert (sig == -1).any()
        assert (sig == 1).any()
        assert (sig == 0).any()

    def test_hub_filter_reduces_rows(self):
        df = _make_oos_df(n_hours=100)  # 4 hubs → 400 rows total
        sig = generate_signals(df, model="linear", hub="HB_NORTH")
        assert len(sig) == 100  # one hub only

    def test_signal_dtype_is_int8(self):
        df = _make_single_hub_oos_df()
        sig = generate_signals(df, model="linear")
        assert sig.dtype == np.int8

    def test_invalid_hub_raises(self):
        df = _make_oos_df()
        with pytest.raises(ValueError, match="HB_FAKE"):
            generate_signals(df, model="linear", hub="HB_FAKE")

    def test_invalid_threshold_raises(self):
        df = _make_oos_df()
        with pytest.raises(ValueError, match="threshold"):
            generate_signals(df, model="linear", threshold=-1.0)

    def test_invalid_model_raises(self):
        df = _make_oos_df()
        with pytest.raises(ValueError, match="spread_pred_fake"):
            generate_signals(df, model="fake")


# ---------------------------------------------------------------------------
# TestComputeHourlyPnl (8 tests)
# ---------------------------------------------------------------------------


class TestComputeHourlyPnl:
    def test_required_columns_present(self):
        df = _make_single_hub_oos_df(n_hours=50)
        sig = generate_signals(df, model="linear")
        result = compute_hourly_pnl(df, sig, model="linear")
        assert _REQUIRED_TRADE_COLS.issubset(set(result.columns))

    def test_zero_pnl_when_all_signals_zero(self):
        df = _make_single_hub_oos_df(n_hours=50)
        # Force all signals to 0 via a threshold larger than any prediction
        sig = generate_signals(df, model="linear", threshold=100.0)
        result = compute_hourly_pnl(df, sig, model="linear")
        assert result["gross_pnl"].sum() == pytest.approx(0.0)
        assert result["net_pnl"].sum() == pytest.approx(0.0)

    def test_gross_pnl_formula_long_position(self):
        # signal=+1, lot=1.0, spread=3.0 → gross_pnl = 3.0 every hour
        df = _make_single_hub_oos_df(n_hours=10, pred_value=5.0, spread_value=3.0)
        sig = generate_signals(df, model="linear", threshold=0.0)
        assert (sig == 1).all()
        result = compute_hourly_pnl(df, sig, lot_mwh=1.0, model="linear")
        np.testing.assert_allclose(result["gross_pnl"].values, 3.0)

    def test_transaction_cost_deducted(self):
        df = _make_single_hub_oos_df(n_hours=10, pred_value=5.0, spread_value=3.0)
        sig = generate_signals(df, model="linear", threshold=0.0)
        result = compute_hourly_pnl(df, sig, lot_mwh=1.0, transaction_cost_per_mwh=1.0, model="linear")
        # net_pnl = gross(3.0) - tc(1.0) = 2.0
        np.testing.assert_allclose(result["net_pnl"].values, 2.0)

    def test_cumulative_monotone_when_all_profitable(self):
        # All long, positive spread, zero TC → cumulative always increases
        df = _make_single_hub_oos_df(n_hours=50, pred_value=5.0, spread_value=10.0)
        sig = generate_signals(df, model="linear")
        result = compute_hourly_pnl(df, sig, lot_mwh=1.0, model="linear")
        cum = result["cumulative_pnl"].values
        assert np.all(np.diff(cum) >= 0.0)

    def test_position_cap_respected(self):
        # lot_mwh=200, max_mwh=100 → positions should not exceed 100
        df = _make_single_hub_oos_df(n_hours=20, pred_value=5.0)
        sig = generate_signals(df, model="linear")
        result = compute_hourly_pnl(df, sig, lot_mwh=200.0, max_mwh=100.0, model="linear")
        assert result["position_mwh"].max() <= 100.0 + 1e-9

    def test_confidence_scaling_varies_position(self):
        # With linearly varying predictions, confidence scaling should produce varying positions
        df = _make_oos_df(n_hours=100, hubs=["HB_NORTH"])
        sig = generate_signals(df, model="linear", threshold=0.0)
        result = compute_hourly_pnl(
            df, sig, lot_mwh=1.0, scale_by_confidence=True, model="linear"
        )
        # Positions should not all be identical (scaling must have happened)
        trading = result[result["signal"] != 0]
        if len(trading) > 1:
            assert trading["position_mwh"].std() > 0.0

    def test_confidence_scaling_requires_model(self):
        df = _make_single_hub_oos_df()
        sig = generate_signals(df, model="linear")
        with pytest.raises(ValueError, match="model"):
            compute_hourly_pnl(df, sig, scale_by_confidence=True, model=None)


# ---------------------------------------------------------------------------
# TestComputePnlMetrics (8 tests)
# ---------------------------------------------------------------------------


class TestComputePnlMetrics:
    def test_all_keys_present(self):
        pnl = pd.Series(np.random.default_rng(0).normal(0, 1, 100))
        result = compute_pnl_metrics(pnl)
        expected_keys = {
            "total_pnl", "annualized_pnl", "sharpe", "sortino",
            "max_drawdown", "win_rate", "cvar_5pct", "n_trades",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_all_zero_pnl_gives_nan_sharpe(self):
        pnl = pd.Series(np.zeros(100))
        result = compute_pnl_metrics(pnl)
        assert math.isnan(result["sharpe"])
        assert result["max_drawdown"] == pytest.approx(0.0)

    def test_varying_pnl_gives_positive_sharpe(self):
        # Linearly increasing P&L: mean > 0, std > 0 → Sharpe > 0
        pnl = pd.Series(np.arange(1.0, 201.0))
        result = compute_pnl_metrics(pnl)
        assert result["sharpe"] > 0.0

    def test_cvar_leq_mean(self):
        rng = np.random.default_rng(42)
        pnl = pd.Series(rng.normal(0, 10, 1000))
        result = compute_pnl_metrics(pnl)
        assert result["cvar_5pct"] <= pnl.mean() + 1e-9

    def test_max_drawdown_nonnegative(self):
        rng = np.random.default_rng(7)
        pnl = pd.Series(rng.normal(0, 1, 500))
        result = compute_pnl_metrics(pnl)
        assert result["max_drawdown"] >= 0.0

    def test_win_rate_in_unit_interval(self):
        rng = np.random.default_rng(99)
        pnl = pd.Series(rng.standard_normal(200))
        result = compute_pnl_metrics(pnl)
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_empty_series_returns_nan(self):
        pnl = pd.Series([], dtype=float)
        result = compute_pnl_metrics(pnl)
        for key in ["total_pnl", "annualized_pnl", "sharpe", "sortino",
                    "max_drawdown", "win_rate", "cvar_5pct"]:
            assert math.isnan(result[key]), f"{key} should be NaN for empty series"
        assert result["n_trades"] == 0

    def test_n_trades_counts_nonzero_hours(self):
        # 50 zeros + 50 nonzero
        arr = np.concatenate([np.zeros(50), np.ones(50)])
        pnl = pd.Series(arr)
        result = compute_pnl_metrics(pnl)
        assert result["n_trades"] == 50


# ---------------------------------------------------------------------------
# TestRunBacktest (8 tests)
# ---------------------------------------------------------------------------


class TestRunBacktest:
    def test_returns_tuple_df_dict(self):
        df = _make_oos_df(n_hours=100, hubs=["HB_NORTH"])
        trade_df, metrics = run_backtest(df, model="linear", hub="HB_NORTH")
        assert isinstance(trade_df, pd.DataFrame)
        assert isinstance(metrics, dict)

    def test_dict_has_all_metric_keys(self):
        df = _make_oos_df(n_hours=100, hubs=["HB_NORTH"])
        _, metrics = run_backtest(df, model="linear", hub="HB_NORTH")
        assert _REQUIRED_METRICS_KEYS.issubset(set(metrics.keys()))

    def test_df_has_all_trade_columns(self):
        df = _make_oos_df(n_hours=100, hubs=["HB_NORTH"])
        trade_df, _ = run_backtest(df, model="linear", hub="HB_NORTH")
        assert _REQUIRED_TRADE_COLS.issubset(set(trade_df.columns))

    def test_trade_rate_in_unit_interval(self):
        df = _make_oos_df(n_hours=200, hubs=["HB_NORTH"])
        _, metrics = run_backtest(df, model="linear", hub="HB_NORTH", threshold=0.0)
        assert 0.0 <= metrics["trade_rate"] <= 1.0

    def test_higher_threshold_fewer_trades(self):
        df = _make_oos_df(n_hours=200, hubs=["HB_NORTH"])
        _, m0 = run_backtest(df, model="linear", hub="HB_NORTH", threshold=0.0)
        _, m50 = run_backtest(df, model="linear", hub="HB_NORTH", threshold=50.0)
        assert m0["n_trades"] >= m50["n_trades"]

    def test_hub_filter_reduces_n_obs(self):
        # 4 hubs → 400 rows; hub="HB_NORTH" → 100 rows
        df = _make_oos_df(n_hours=100)
        _, m_all = run_backtest(df, model="linear", hub=None)
        _, m_hub = run_backtest(df, model="linear", hub="HB_NORTH")
        assert m_all["n_obs"] == 4 * m_hub["n_obs"]

    def test_total_pnl_matches_net_pnl_sum(self):
        df = _make_oos_df(n_hours=100, hubs=["HB_NORTH"])
        trade_df, metrics = run_backtest(df, model="linear", hub="HB_NORTH")
        assert metrics["total_pnl"] == pytest.approx(trade_df["net_pnl"].sum())

    def test_invalid_model_raises(self):
        df = _make_oos_df(n_hours=50, hubs=["HB_NORTH"])
        with pytest.raises(ValueError, match="spread_pred_fake_model"):
            run_backtest(df, model="fake_model", hub="HB_NORTH")


# ---------------------------------------------------------------------------
# TestRunAllBacktests (6 tests)
# ---------------------------------------------------------------------------


class TestRunAllBacktests:
    def test_row_count_per_hub(self):
        df = _make_oos_df(n_hours=100)
        models = ["linear", "ridge"]
        thresholds = [0.0, 5.0, 10.0]
        result = run_all_backtests(df, models=models, thresholds=thresholds, per_hub=True)
        # 2 models × 4 hubs × 3 thresholds = 24
        assert len(result) == 24

    def test_column_schema(self):
        df = _make_oos_df(n_hours=50, hubs=["HB_NORTH"])
        result = run_all_backtests(df, models=["linear"], thresholds=[0.0], per_hub=True)
        assert _REQUIRED_METRICS_KEYS.issubset(set(result.columns))

    def test_per_hub_false_aggregates_all(self):
        df = _make_oos_df(n_hours=50)
        result = run_all_backtests(
            df, models=["linear", "lasso"], thresholds=[0.0, 5.0], per_hub=False
        )
        # 2 models × 1 × 2 thresholds = 4 rows
        assert len(result) == 4
        assert (result["hub"] == "all").all()

    def test_higher_threshold_monotone_fewer_trades(self):
        df = _make_oos_df(n_hours=200, hubs=["HB_NORTH"])
        thresholds = [0.0, 5.0, 15.0]
        result = run_all_backtests(
            df, models=["linear"], thresholds=thresholds, per_hub=True
        )
        north = result[result["hub"] == "HB_NORTH"].sort_values("threshold")
        trades = north["n_trades"].values
        # n_trades should be non-increasing as threshold increases
        assert all(trades[i] >= trades[i + 1] for i in range(len(trades) - 1))

    def test_model_column_contains_all_models(self):
        df = _make_oos_df(n_hours=50)
        models = ["linear", "lgbm"]
        result = run_all_backtests(df, models=models, thresholds=[0.0], per_hub=False)
        assert set(result["model"].unique()) == set(models)

    def test_returns_dataframe(self):
        df = _make_oos_df(n_hours=50, hubs=["HB_NORTH"])
        result = run_all_backtests(df, models=["linear"], thresholds=[0.0], per_hub=True)
        assert isinstance(result, pd.DataFrame)
