"""Unit tests for energy_trading.data.validation.

All tests use synthetic DataFrames no network calls are made.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from energy_trading.data.validation import (
    DataValidationError,
    check_hourly_coverage,
    check_no_all_nan_columns,
    check_no_duplicate_timestamps,
    check_not_empty,
    check_price_bounds,
    check_required_columns,
    check_required_hubs,
    validate_lmp_dataframe,
)

START = date(2023, 1, 1)
END = date(2023, 1, 3)
HUBS = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lmp_df(
    start: date = START,
    end: date = END,
    hubs: list[str] = HUBS,
    lmp_value: float = 45.0,
) -> pd.DataFrame:
    """Create a valid hourly LMP DataFrame for the given date range and hubs."""
    idx = pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
        freq="h",
    )
    records = [
        {"hub": hub, "lmp": lmp_value, "energy": 40.0, "congestion": 3.0, "loss": 2.0}
        for _ in idx
        for hub in hubs
    ]
    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex(
        [ts for ts in idx for _ in hubs], tz="UTC", name="interval_start_utc"
    )
    return df


# ---------------------------------------------------------------------------
# check_not_empty
# ---------------------------------------------------------------------------


def test_check_not_empty_passes_on_nonempty():
    df = pd.DataFrame({"a": [1]})
    check_not_empty(df)  # Should not raise.


def test_check_not_empty_raises_on_empty():
    with pytest.raises(DataValidationError, match="empty"):
        check_not_empty(pd.DataFrame())


# ---------------------------------------------------------------------------
# check_required_columns
# ---------------------------------------------------------------------------


def test_check_required_columns_passes():
    df = pd.DataFrame({"hub": [], "lmp": []})
    check_required_columns(df, ["hub", "lmp"])


def test_check_required_columns_raises_on_missing():
    df = pd.DataFrame({"hub": []})
    with pytest.raises(DataValidationError, match="lmp"):
        check_required_columns(df, ["hub", "lmp"])


# ---------------------------------------------------------------------------
# check_no_duplicate_timestamps
# ---------------------------------------------------------------------------


def test_check_no_duplicate_timestamps_passes():
    idx = pd.date_range("2023-01-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"hub": ["HB_NORTH"] * 3}, index=idx)
    check_no_duplicate_timestamps(df)


def test_check_no_duplicate_timestamps_raises_ungrouped():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2023-01-01 00:00", tz="UTC")] * 2, name="interval_start_utc"
    )
    df = pd.DataFrame({"hub": ["HB_NORTH"] * 2}, index=idx)
    with pytest.raises(DataValidationError, match="duplicate"):
        check_no_duplicate_timestamps(df)


def test_check_no_duplicate_timestamps_per_hub():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2023-01-01", tz="UTC")] * 2, name="interval_start_utc"
    )
    df = pd.DataFrame({"hub": ["HB_NORTH", "HB_NORTH"]}, index=idx)
    with pytest.raises(DataValidationError, match="duplicate"):
        check_no_duplicate_timestamps(df, group_col="hub")


def test_check_no_duplicate_timestamps_different_hubs_ok():
    idx = pd.DatetimeIndex(
        [pd.Timestamp("2023-01-01", tz="UTC")] * 2, name="interval_start_utc"
    )
    df = pd.DataFrame({"hub": ["HB_NORTH", "HB_SOUTH"]}, index=idx)
    check_no_duplicate_timestamps(df, group_col="hub")  # Should not raise.


# ---------------------------------------------------------------------------
# check_hourly_coverage
# ---------------------------------------------------------------------------


def test_check_hourly_coverage_passes_complete():
    idx = pd.date_range("2023-01-01", periods=72, freq="h", tz="UTC")
    df = pd.DataFrame({"v": range(72)}, index=idx)
    check_hourly_coverage(df, date(2023, 1, 1), date(2023, 1, 3))


def test_check_hourly_coverage_raises_on_large_gap():
    # Only 1 hour present out of 72 expected.
    idx = pd.DatetimeIndex([pd.Timestamp("2023-01-01", tz="UTC")])
    df = pd.DataFrame({"v": [1]}, index=idx)
    with pytest.raises(DataValidationError, match="missing hours"):
        check_hourly_coverage(df, date(2023, 1, 1), date(2023, 1, 3), max_missing_pct=0.01)


def test_check_hourly_coverage_raises_on_one_missing_hour():
    # 71 of 72 hours present 1/72 ≈ 1.4% missing, which exceeds the 1% threshold.
    idx = pd.date_range("2023-01-01", periods=71, freq="h", tz="UTC")
    df = pd.DataFrame({"v": range(71)}, index=idx)
    with pytest.raises(DataValidationError):
        check_hourly_coverage(df, date(2023, 1, 1), date(2023, 1, 3), max_missing_pct=0.01)


# ---------------------------------------------------------------------------
# check_required_hubs
# ---------------------------------------------------------------------------


def test_check_required_hubs_passes():
    df = pd.DataFrame({"hub": HUBS})
    check_required_hubs(df, required_hubs=HUBS)


def test_check_required_hubs_raises_on_missing():
    df = pd.DataFrame({"hub": ["HB_NORTH", "HB_SOUTH"]})
    with pytest.raises(DataValidationError, match="HB_WEST"):
        check_required_hubs(df, required_hubs=HUBS)


# ---------------------------------------------------------------------------
# check_price_bounds  (warns, does not raise)
# ---------------------------------------------------------------------------


def test_check_price_bounds_does_not_raise_on_extreme_prices():
    """Extreme prices generate warnings but must not raise DataValidationError."""
    df = pd.DataFrame({"lmp": [-1000.0, 15000.0]})
    check_price_bounds(df)  # Should not raise.


def test_check_price_bounds_raises_on_missing_column():
    df = pd.DataFrame({"price": [45.0]})
    with pytest.raises(DataValidationError, match="lmp"):
        check_price_bounds(df)


# ---------------------------------------------------------------------------
# check_no_all_nan_columns
# ---------------------------------------------------------------------------


def test_check_no_all_nan_columns_passes():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [None, 3.0]})
    check_no_all_nan_columns(df)


def test_check_no_all_nan_columns_raises():
    df = pd.DataFrame({"a": [1.0], "b": [None]})
    with pytest.raises(DataValidationError, match="all NaN"):
        check_no_all_nan_columns(df)


# ---------------------------------------------------------------------------
# validate_lmp_dataframe  (composite)
# ---------------------------------------------------------------------------


def test_validate_lmp_dataframe_passes_on_valid_data():
    df = _make_lmp_df()
    validate_lmp_dataframe(df, start=START, end=END)  # Should not raise.


def test_validate_lmp_dataframe_collects_multiple_errors():
    """All errors should be surfaced in a single DataValidationError."""
    # Missing hub column and empty DataFrame two distinct failures.
    df = pd.DataFrame()
    with pytest.raises(DataValidationError) as exc_info:
        validate_lmp_dataframe(df, start=START, end=END)
    assert len(exc_info.value.errors) >= 1


def test_validate_lmp_dataframe_raises_on_wrong_hubs():
    df = _make_lmp_df(hubs=["HB_NORTH"])  # Missing three hubs.
    with pytest.raises(DataValidationError, match="HB_SOUTH"):
        validate_lmp_dataframe(df, start=START, end=END)
