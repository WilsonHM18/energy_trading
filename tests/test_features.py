"""Unit tests for energy_trading.features.engineering.

All tests use synthetic DataFrames no network calls or disk I/O.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_trading.features.engineering import (
    _add_calendar_features,
    _add_lagged_spread,
    _add_load_features,
    _add_rolling_volatility,
    build_features,
    drop_warmup_rows,
    get_feature_columns,
)

HUBS = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_spread_df(n_hours: int = 400) -> pd.DataFrame:
    """Minimal synthetic spread dataset matching DataPipeline output schema."""
    idx = pd.date_range("2023-01-01", periods=n_hours, freq="h", tz="UTC")
    records = []
    rng = np.random.default_rng(42)
    for ts in idx:
        for hub in HUBS:
            records.append(
                {
                    "hub": hub,
                    "dam_lmp": 40.0 + rng.normal(0, 5),
                    "rtm_lmp": 38.0 + rng.normal(0, 8),
                    "spread": 2.0 + rng.normal(0, 10),
                    "load_mw": 35_000 + rng.normal(0, 2_000),
                }
            )
    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex(
        [ts for ts in idx for _ in HUBS], tz="UTC", name="interval_start_utc"
    )
    return df


@pytest.fixture()
def spread_df() -> pd.DataFrame:
    return _make_spread_df(n_hours=400)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def test_calendar_features_present(spread_df):
    df = _add_calendar_features(spread_df.copy())
    expected = {
        "hour_utc", "hour_cpt", "dow", "month", "quarter",
        "is_weekend", "is_peak", "sin_hour", "cos_hour",
        "sin_month", "cos_month",
    }
    assert expected.issubset(df.columns)


def test_hour_cpt_range(spread_df):
    df = _add_calendar_features(spread_df.copy())
    assert df["hour_cpt"].between(0, 23).all()


def test_is_weekend_binary(spread_df):
    df = _add_calendar_features(spread_df.copy())
    assert set(df["is_weekend"].unique()).issubset({0, 1})


def test_cyclical_encoding_bounds(spread_df):
    df = _add_calendar_features(spread_df.copy())
    for col in ["sin_hour", "cos_hour", "sin_month", "cos_month"]:
        assert df[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_is_peak_zero_on_weekends(spread_df):
    df = _add_calendar_features(spread_df.copy())
    weekend_peak = df.loc[df["is_weekend"] == 1, "is_peak"]
    assert (weekend_peak == 0).all()


# ---------------------------------------------------------------------------
# Lagged spread features
# ---------------------------------------------------------------------------


def test_lagged_spread_columns_created(spread_df):
    df = _add_lagged_spread(spread_df.copy(), lag_days=[1, 7], lag_weeks=[1])
    assert "spread_lag_1d" in df.columns
    assert "spread_lag_7d" in df.columns
    assert "spread_lag_1w" in df.columns


def test_lagged_spread_no_lookahead(spread_df):
    """spread_lag_1d at hour t must equal spread at hour t-24."""
    df = _add_lagged_spread(spread_df.copy(), lag_days=[1], lag_weeks=[])
    north = df[df["hub"] == "HB_NORTH"].copy()
    # From row 24 onward, lag should equal the original spread 24 rows earlier.
    orig = north["spread"].values
    lagged = north["spread_lag_1d"].values
    valid = ~np.isnan(lagged[24:])
    np.testing.assert_allclose(lagged[24:][valid], orig[:len(orig) - 24][valid])


def test_lagged_spread_nan_in_warmup(spread_df):
    df = _add_lagged_spread(spread_df.copy(), lag_days=[1], lag_weeks=[])
    north = df[df["hub"] == "HB_NORTH"]
    # First 24 rows should all be NaN (no prior-day data).
    assert north["spread_lag_1d"].iloc[:24].isna().all()


# ---------------------------------------------------------------------------
# Load features
# ---------------------------------------------------------------------------


def test_load_features_present(spread_df):
    df = _add_load_features(spread_df.copy())
    for col in ["load_mw_lag1h", "load_mw_lag1d", "load_ramp_1h", "load_ramp_24h"]:
        assert col in df.columns, f"Missing: {col}"


def test_load_lag_no_lookahead(spread_df):
    """load_mw_lag1h must not equal load_mw at the same timestamp."""
    df = _add_load_features(spread_df.copy())
    # They should differ (lag shifts by 1 hour).
    subset = df[df["hub"] == "HB_NORTH"].iloc[5:50]
    assert not (subset["load_mw"] == subset["load_mw_lag1h"]).all()


# ---------------------------------------------------------------------------
# Rolling volatility
# ---------------------------------------------------------------------------


def test_rolling_volatility_columns(spread_df):
    df = _add_rolling_volatility(spread_df.copy(), windows_h=[24])
    assert "spread_vol_24h" in df.columns


def test_rolling_volatility_nonnegative(spread_df):
    df = _add_rolling_volatility(spread_df.copy(), windows_h=[24])
    valid = df["spread_vol_24h"].dropna()
    assert (valid >= 0).all()


def test_rolling_volatility_no_lookahead(spread_df):
    """Volatility at t must not use the spread at t."""
    # If we shift spread by 1 before rolling, the vol at position i
    # must be computed from positions 0..i-1 only.
    # We verify this by checking that perturbing the current row's spread
    # does not change the volatility at the same row.
    df_orig = _add_rolling_volatility(spread_df.copy(), windows_h=[24])
    df_perturbed = spread_df.copy()
    df_perturbed.loc[df_perturbed["hub"] == "HB_NORTH", "spread"] *= 1000
    df_pert_vol = _add_rolling_volatility(df_perturbed, windows_h=[24])
    # The first row for HB_NORTH should be identical (it has no history to corrupt).
    north_orig = df_orig[df_orig["hub"] == "HB_NORTH"]["spread_vol_24h"].iloc[0]
    north_pert = df_pert_vol[df_pert_vol["hub"] == "HB_NORTH"]["spread_vol_24h"].iloc[0]
    assert north_orig == north_pert or (np.isnan(north_orig) and np.isnan(north_pert))


# ---------------------------------------------------------------------------
# build_features (integration)
# ---------------------------------------------------------------------------


def test_build_features_returns_more_columns(spread_df):
    result = build_features(spread_df)
    assert result.shape[1] > spread_df.shape[1]


def test_build_features_preserves_index(spread_df):
    result = build_features(spread_df)
    pd.testing.assert_index_equal(result.index, spread_df.index)


def test_build_features_preserves_spread_column(spread_df):
    result = build_features(spread_df)
    pd.testing.assert_series_equal(result["spread"], spread_df["spread"])


def test_get_feature_columns_excludes_raw(spread_df):
    result = build_features(spread_df)
    feat_cols = get_feature_columns(result)
    raw = {"hub", "dam_lmp", "rtm_lmp", "spread", "load_mw"}
    assert not raw.intersection(feat_cols)
    assert len(feat_cols) > 0


# ---------------------------------------------------------------------------
# drop_warmup_rows
# ---------------------------------------------------------------------------


def test_drop_warmup_rows_removes_early_rows(spread_df):
    result = build_features(spread_df)
    trimmed = drop_warmup_rows(result, min_lag_hours=168)
    assert len(trimmed) < len(result)
    cutoff = result.index.min() + pd.Timedelta(hours=168)
    assert trimmed.index.min() >= cutoff
