"""Tests for src/energy_trading/backtest/risk.py.

All tests are synthetic (no disk I/O).  Helpers produce minimal DataFrames
that satisfy the schemas expected by each risk function.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from energy_trading.backtest.risk import (
    compute_regime_metrics,
    compute_rolling_sharpe,
    compute_tail_risk,
    identify_drawdown_periods,
    parameter_robustness,
    tc_sensitivity,
)

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_UTC = "UTC"
_HUB = "HB_NORTH"


def _make_index(n_hours: int, start: str = "2021-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_hours, freq="h", tz=_UTC, name="interval_start_utc")


def _make_cumulative_pnl(values: list[float]) -> pd.Series:
    """Create a cumulative P&L Series with a UTC DatetimeIndex."""
    idx = _make_index(len(values))
    return pd.Series(values, index=idx, name="cumulative_pnl", dtype=np.float64)


def _make_trade_df(
    n_hours: int = 1000,
    net_pnl_val: float = 3.0,
    start: str = "2021-01-01",
) -> pd.DataFrame:
    """Minimal trade_df compatible with compute_hourly_pnl output schema."""
    idx = _make_index(n_hours, start)
    net_pnl = np.full(n_hours, net_pnl_val, dtype=np.float64)
    return pd.DataFrame(
        {
            "signal": np.ones(n_hours, dtype=np.int8),
            "position_mwh": np.ones(n_hours, dtype=np.float64),
            "spread_pred": net_pnl,
            "spread_actual": net_pnl,
            "gross_pnl": net_pnl,
            "transaction_cost_paid": np.zeros(n_hours, dtype=np.float64),
            "net_pnl": net_pnl,
            "cumulative_pnl": np.cumsum(net_pnl),
        },
        index=idx,
    )


def _make_features_df(
    n_hours: int = 1000,
    hub: str = _HUB,
    with_eia: bool = True,
    start: str = "2021-01-01",
) -> pd.DataFrame:
    """Minimal features_df compatible with compute_regime_metrics."""
    idx = _make_index(n_hours, start)
    rng = np.random.default_rng(42)
    data: dict = {
        "hub": hub,
        "is_peak": np.tile([0, 1], math.ceil(n_hours / 2))[:n_hours].astype(np.int8),
        "spread_vol_24h": rng.uniform(5.0, 50.0, n_hours),
    }
    if with_eia:
        data["wind_actual_mw"] = rng.uniform(5000.0, 20000.0, n_hours)
        data["gas_price_mmbtu"] = rng.uniform(2.0, 8.0, n_hours)
    return pd.DataFrame(data, index=idx)


def _make_oos_df(
    n_hours: int = 200,
    hub: str = _HUB,
    pred_value: float = 5.0,
    spread_value: float = 3.0,
    start: str = "2021-01-01",
) -> pd.DataFrame:
    """Minimal single-hub OOS DataFrame for tc_sensitivity / parameter_robustness."""
    idx = _make_index(n_hours, start)
    return pd.DataFrame(
        {
            "hub": hub,
            "spread": np.full(n_hours, spread_value, dtype=np.float64),
            "spread_pred_linear": np.full(n_hours, pred_value, dtype=np.float64),
            "spread_pred_ridge": np.full(n_hours, pred_value, dtype=np.float64),
            "spread_pred_lasso": np.full(n_hours, pred_value, dtype=np.float64),
            "spread_pred_lgbm": np.full(n_hours, pred_value, dtype=np.float64),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# TestIdentifyDrawdownPeriods
# ---------------------------------------------------------------------------


class TestIdentifyDrawdownPeriods:
    def test_no_drawdown_monotone_increasing(self):
        """Monotone-increasing cumulative P&L has no drawdowns."""
        values = list(range(100))
        result = identify_drawdown_periods(_make_cumulative_pnl(values))
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert set(result.columns) >= {"depth_dollars", "start_date", "trough_date"}

    def test_single_drawdown_depth_correct(self):
        """Single V-shaped drawdown: depth is correctly measured."""
        # Peak=10, trough=3, recovery=10 → depth=7
        values = [0, 5, 10, 7, 3, 6, 10, 12]
        result = identify_drawdown_periods(_make_cumulative_pnl(values))
        assert len(result) == 1
        assert result["depth_dollars"].iloc[0] == pytest.approx(7.0)

    def test_multiple_drawdowns_sorted_by_depth(self):
        """Multiple drawdowns are returned sorted worst-first."""
        # Two drawdowns: deep (drop 8) and shallow (drop 2)
        values = [0, 10, 2, 10, 12, 10, 12]
        result = identify_drawdown_periods(_make_cumulative_pnl(values))
        assert len(result) >= 1
        depths = result["depth_dollars"].tolist()
        assert depths == sorted(depths, reverse=True)

    def test_recovered_drawdown_flag(self):
        """A drawdown that recovers has recovered=True and finite recovery_hours."""
        values = [0, 10, 5, 10, 15]
        result = identify_drawdown_periods(_make_cumulative_pnl(values))
        assert len(result) >= 1
        rec_row = result.iloc[0]
        assert rec_row["recovered"] is True or rec_row["recovered"] == True  # noqa: E712
        assert not math.isnan(rec_row["recovery_hours"])
        assert rec_row["recovery_hours"] >= 0

    def test_unrecovered_drawdown_flag(self):
        """A drawdown that never recovers has recovered=False and NaN recovery_hours."""
        # Peaks at 10, drops to 5 and never returns
        values = [0, 5, 10, 7, 5, 4]
        result = identify_drawdown_periods(_make_cumulative_pnl(values))
        assert len(result) >= 1
        # The last drawdown (still underwater at end) should be unrecovered
        unrecovered = result[~result["recovered"]]
        assert len(unrecovered) >= 1
        assert all(math.isnan(v) for v in unrecovered["recovery_hours"])

    def test_empty_series_returns_empty_df(self):
        """Empty input returns empty DataFrame with correct columns."""
        empty = pd.Series([], dtype=np.float64)
        result = identify_drawdown_periods(empty)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert "depth_dollars" in result.columns


# ---------------------------------------------------------------------------
# TestComputeRegimeMetrics
# ---------------------------------------------------------------------------


class TestComputeRegimeMetrics:
    def test_returns_dataframe_with_regime_column(self):
        trade = _make_trade_df(n_hours=2000)
        features = _make_features_df(n_hours=2000)
        result = compute_regime_metrics(trade, features, _HUB)
        assert isinstance(result, pd.DataFrame)
        assert "regime" in result.columns

    def test_core_regimes_all_present(self):
        """peak, off_peak, high_vol, low_vol, all must always appear."""
        trade = _make_trade_df(n_hours=2000)
        features = _make_features_df(n_hours=2000)
        result = compute_regime_metrics(trade, features, _HUB)
        regimes = set(result["regime"].tolist())
        for expected in ("all", "peak", "off_peak", "high_vol", "low_vol"):
            assert expected in regimes, f"'{expected}' regime missing from result"

    def test_metrics_columns_present(self):
        """All expected metric columns are present."""
        trade = _make_trade_df(n_hours=1000)
        features = _make_features_df(n_hours=1000)
        result = compute_regime_metrics(trade, features, _HUB)
        for col in ("total_pnl", "sharpe", "n_obs", "n_trades", "win_rate"):
            assert col in result.columns, f"'{col}' missing from regime metrics"

    def test_peak_offpeak_partition_sums_to_all(self):
        """peak n_obs + off_peak n_obs == all n_obs."""
        trade = _make_trade_df(n_hours=1000)
        features = _make_features_df(n_hours=1000)
        result = compute_regime_metrics(trade, features, _HUB)
        df = result.set_index("regime")
        assert df.loc["peak", "n_obs"] + df.loc["off_peak", "n_obs"] == df.loc["all", "n_obs"]

    def test_missing_wind_column_no_error(self):
        """Missing wind_actual_mw column does not raise an error."""
        trade = _make_trade_df(n_hours=500)
        features = _make_features_df(n_hours=500, with_eia=False)
        # Should complete without raising
        result = compute_regime_metrics(trade, features, _HUB)
        assert isinstance(result, pd.DataFrame)
        # wind regimes should not appear
        assert "high_wind" not in result["regime"].values
        assert "low_wind" not in result["regime"].values

    def test_uri_regime_isolated_to_feb_2021(self):
        """uri_storm regime only includes rows in [2021-02-10, 2021-02-21) UTC."""
        # Create a trade_df that spans Feb 2021
        n = 24 * 30  # 30 days in Feb area
        trade = _make_trade_df(n_hours=n, start="2021-02-01")
        features = _make_features_df(n_hours=n, start="2021-02-01")
        result = compute_regime_metrics(trade, features, _HUB)
        if "uri_storm" in result["regime"].values:
            uri_n = int(result.loc[result["regime"] == "uri_storm", "n_obs"].iloc[0])
            # Uri window is 11 days × 24 hours = 264 hours max
            assert uri_n <= 264


# ---------------------------------------------------------------------------
# TestComputeTailRisk
# ---------------------------------------------------------------------------


class TestComputeTailRisk:
    def test_all_keys_present(self):
        pnl = pd.Series(np.random.default_rng(0).normal(1.0, 5.0, 1000))
        result = compute_tail_risk(pnl)
        expected_keys = {
            "var_01pct", "var_05pct", "var_10pct",
            "cvar_01pct", "cvar_05pct", "cvar_10pct",
            "skewness", "excess_kurtosis", "max_consecutive_loss_hours",
        }
        assert expected_keys <= set(result.keys())

    def test_var_ordering(self):
        """Stricter confidence → more extreme (lower) VaR."""
        pnl = pd.Series(np.random.default_rng(1).normal(0, 10.0, 5000))
        result = compute_tail_risk(pnl)
        assert result["var_01pct"] <= result["var_05pct"] <= result["var_10pct"]

    def test_cvar_le_var(self):
        """CVaR is always <= VaR (CVaR is the expected loss beyond VaR)."""
        pnl = pd.Series(np.random.default_rng(2).normal(0, 10.0, 5000))
        result = compute_tail_risk(pnl)
        for lvl in ("01", "05", "10"):
            assert result[f"cvar_{lvl}pct"] <= result[f"var_{lvl}pct"]

    def test_symmetric_distribution_skewness_near_zero(self):
        """Normal distribution has skewness ≈ 0."""
        rng = np.random.default_rng(3)
        pnl = pd.Series(rng.normal(0, 1.0, 10_000))
        result = compute_tail_risk(pnl)
        assert abs(result["skewness"]) < 0.1

    def test_empty_series_all_nan(self):
        result = compute_tail_risk(pd.Series([], dtype=np.float64))
        for key in ("var_01pct", "var_05pct", "skewness", "excess_kurtosis"):
            assert math.isnan(result[key])
        assert result["max_consecutive_loss_hours"] == 0.0

    def test_consecutive_losses_counted_correctly(self):
        """Longest run of negative P&L is correctly identified."""
        # 3 positive, 5 negative, 2 positive, 1 negative
        values = [1.0] * 3 + [-1.0] * 5 + [1.0] * 2 + [-1.0] * 1
        result = compute_tail_risk(pd.Series(values))
        assert result["max_consecutive_loss_hours"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# TestTcSensitivity
# ---------------------------------------------------------------------------


class TestTcSensitivity:
    def test_returns_dataframe_with_tc_column(self):
        oos = _make_oos_df()
        result = tc_sensitivity(oos, "lgbm", _HUB, [0.0, 0.5, 1.0])
        assert isinstance(result, pd.DataFrame)
        assert "tc" in result.columns

    def test_row_count_equals_tc_values_length(self):
        oos = _make_oos_df()
        tc_vals = [0.0, 0.25, 0.50, 1.00, 2.00]
        result = tc_sensitivity(oos, "lgbm", _HUB, tc_vals)
        assert len(result) == len(tc_vals)

    def test_higher_tc_lower_total_pnl(self):
        """For a positive-expectation strategy, TC should monotonically reduce P&L."""
        oos = _make_oos_df(pred_value=5.0, spread_value=5.0)  # always-profitable
        tc_vals = [0.0, 0.5, 1.0, 2.0]
        result = tc_sensitivity(oos, "lgbm", _HUB, tc_vals)
        pnls = result.sort_values("tc")["total_pnl"].tolist()
        for i in range(len(pnls) - 1):
            assert pnls[i] >= pnls[i + 1], (
                f"total_pnl not non-increasing: tc idx {i} pnl={pnls[i]:.2f} "
                f"> tc idx {i+1} pnl={pnls[i+1]:.2f}"
            )

    def test_tc_zero_matches_run_backtest(self):
        """tc=0 result matches run_backtest directly."""
        from energy_trading.backtest.engine import run_backtest

        oos = _make_oos_df()
        result_df = tc_sensitivity(oos, "lgbm", _HUB, [0.0])
        _, metrics = run_backtest(oos, "lgbm", hub=_HUB, transaction_cost_per_mwh=0.0)
        assert result_df["total_pnl"].iloc[0] == pytest.approx(metrics["total_pnl"])

    def test_sorted_by_tc_ascending(self):
        oos = _make_oos_df()
        tc_vals = [2.0, 0.5, 0.0, 1.0]
        result = tc_sensitivity(oos, "lgbm", _HUB, tc_vals)
        assert list(result["tc"]) == sorted(tc_vals)


# ---------------------------------------------------------------------------
# TestComputeRollingSharpe
# ---------------------------------------------------------------------------


class TestComputeRollingSharpe:
    def test_returns_series_same_index(self):
        pnl = pd.Series(
            np.random.default_rng(10).normal(0.5, 2.0, 1000),
            index=_make_index(1000),
        )
        result = compute_rolling_sharpe(pnl, window_hours=100)
        assert isinstance(result, pd.Series)
        assert len(result) == len(pnl)
        assert result.index.equals(pnl.index)

    def test_first_entries_are_nan(self):
        n = 500
        window = 100
        pnl = pd.Series(np.ones(n), index=_make_index(n))
        result = compute_rolling_sharpe(pnl, window_hours=window)
        # First window-1 entries must be NaN
        assert result.iloc[: window - 1].isna().all()

    def test_constant_pnl_gives_nan_sharpe(self):
        """Constant P&L has std=0 → rolling Sharpe must be NaN."""
        n = 300
        window = 50
        pnl = pd.Series(np.ones(n) * 5.0, index=_make_index(n))
        result = compute_rolling_sharpe(pnl, window_hours=window)
        valid = result.dropna()
        assert len(valid) == 0, "Expected all NaN for constant P&L"

    def test_positive_pnl_positive_rolling_sharpe(self):
        """Consistently positive P&L with variance should give positive Sharpe."""
        rng = np.random.default_rng(20)
        n = 2000
        window = 200
        pnl = pd.Series(rng.normal(2.0, 1.0, n), index=_make_index(n))
        result = compute_rolling_sharpe(pnl, window_hours=window)
        valid = result.dropna()
        assert (valid > 0).all(), "Expected all-positive Sharpe for positive-drift series"

    def test_name_is_rolling_sharpe(self):
        pnl = pd.Series(np.ones(100), index=_make_index(100))
        result = compute_rolling_sharpe(pnl)
        assert result.name == "rolling_sharpe"


# ---------------------------------------------------------------------------
# TestParameterRobustness
# ---------------------------------------------------------------------------


class TestParameterRobustness:
    def test_returns_dataframe(self):
        oos = _make_oos_df()
        result = parameter_robustness(oos, "lgbm", _HUB, thresholds=[0.0, 5.0])
        assert isinstance(result, pd.DataFrame)

    def test_row_count_correct(self):
        oos = _make_oos_df()
        thresholds = [0.0, 2.0, 5.0, 10.0]
        lot_sizes = [1.0, 2.0]
        result = parameter_robustness(oos, "lgbm", _HUB, thresholds=thresholds, lot_sizes=lot_sizes)
        assert len(result) == len(thresholds) * len(lot_sizes)

    def test_higher_threshold_lower_trade_rate(self):
        """Higher threshold → equal or fewer trades (monotone non-increasing)."""
        oos = _make_oos_df(n_hours=400, pred_value=5.0)
        # Thresholds below and above the pred value
        thresholds = [0.0, 2.0, 4.0, 6.0, 10.0]
        result = parameter_robustness(oos, "lgbm", _HUB, thresholds=thresholds)
        rates = result.sort_values("threshold")["trade_rate"].tolist()
        for i in range(len(rates) - 1):
            assert rates[i] >= rates[i + 1], (
                f"trade_rate not non-increasing: threshold idx {i} "
                f"rate={rates[i]:.4f} > idx {i+1} rate={rates[i+1]:.4f}"
            )

    def test_columns_include_threshold_and_lot_mwh(self):
        oos = _make_oos_df()
        result = parameter_robustness(oos, "lgbm", _HUB, thresholds=[0.0, 5.0])
        assert "threshold" in result.columns
        assert "lot_mwh" in result.columns
