"""Shared pytest fixtures for the energy-trading test suite."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Date range constants
# ---------------------------------------------------------------------------
START = date(2023, 1, 1)
END = date(2023, 1, 7)


# ---------------------------------------------------------------------------
# LMP fixtures
# ---------------------------------------------------------------------------

HUBS = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]


def _make_hourly_index(start: date, end: date) -> pd.DatetimeIndex:
    return pd.date_range(
        start=pd.Timestamp(start, tz="UTC"),
        end=pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23),
        freq="h",
    )


@pytest.fixture()
def hourly_index() -> pd.DatetimeIndex:
    """Complete 7-day UTC hourly index (168 timestamps)."""
    return _make_hourly_index(START, END)


@pytest.fixture()
def valid_dam_df(hourly_index) -> pd.DataFrame:
    """Minimal valid DAM LMP DataFrame for all four ERCOT hubs."""
    records = []
    for ts in hourly_index:
        for hub in HUBS:
            records.append(
                {
                    "hub": hub,
                    "lmp": 35.0 + hash((ts, hub)) % 20,
                    "energy": 30.0,
                    "congestion": 3.0,
                    "loss": 2.0,
                }
            )
    df = pd.DataFrame.from_records(records)
    df.index = pd.DatetimeIndex([r[0] for r in [(ts, hub) for ts in hourly_index for hub in HUBS]])
    df.index.name = "interval_start_utc"
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
    return df
