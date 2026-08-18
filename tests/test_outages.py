"""Unit tests for energy_trading.data.outages (ERCOTOutageClient).

All tests use mocked gridstatus responses no live network calls.

Historical-coverage tests mock ``ERCOTOutageClient._get_now`` to set a
deterministic "current time" so the 31-day cutoff is stable regardless of
when the test suite is run.
"""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_trading.data.outages import (
    ERCOTOutageClient,
    OutageDataError,
    _EXPECTED_COLS,
    _HISTORICAL_COVERAGE_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fixed_now() -> pd.Timestamp:
    """Return a fixed UTC 'now' for deterministic cutoff tests."""
    return pd.Timestamp("2024-06-15 12:00:00", tz="UTC")


def _make_mock_outage_df(n_hours: int = 24, start_str: str = "2024-06-08") -> pd.DataFrame:
    """Build a fake gridstatus outage DataFrame (US/Central DatetimeIndex)."""
    idx = pd.date_range(start=start_str, periods=n_hours, freq="h", tz="US/Central")
    return pd.DataFrame(
        {
            "Total Resource MW Zone North": [1000.0 + i for i in range(n_hours)],
            "Total Resource MW Zone South": [500.0 + i for i in range(n_hours)],
            "Total Resource MW Zone West": [800.0 + i for i in range(n_hours)],
            "Total Resource MW Zone Houston": [300.0 + i for i in range(n_hours)],
            "Total Resource MW": [2600.0 + 4 * i for i in range(n_hours)],
            "Total IRR MW": [0.0] * n_hours,
        },
        index=idx,
    )


def _make_aggregate_mock_df(n_hours: int = 24, start_str: str = "2024-06-08") -> pd.DataFrame:
    """Build a fake gridstatus outage DataFrame with aggregate columns only (older format)."""
    idx = pd.date_range(start=start_str, periods=n_hours, freq="h", tz="US/Central")
    return pd.DataFrame(
        {
            "Total Resource MW": [5000.0 + i for i in range(n_hours)],
            "Total IRR MW": [0.0] * n_hours,
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestERCOTOutageClient:
    def test_get_hourly_outage_capacity_returns_dataframe(self):
        """get_hourly_outage_capacity returns a pd.DataFrame."""
        client = ERCOTOutageClient()
        mock_df = _make_mock_outage_df()

        with patch.object(client, "_get_now", return_value=_fixed_now()):
            with patch.object(client._backend, "get_hourly_resource_outage_capacity",
                              return_value=mock_df):
                result = client.get_hourly_outage_capacity(
                    date(2024, 6, 8), date(2024, 6, 8)
                )

        assert isinstance(result, pd.DataFrame)

    def test_index_is_utc_datetimeindex(self):
        """Result index is a UTC-aware DatetimeIndex named interval_start_utc."""
        client = ERCOTOutageClient()
        mock_df = _make_mock_outage_df()

        with patch.object(client, "_get_now", return_value=_fixed_now()):
            with patch.object(client._backend, "get_hourly_resource_outage_capacity",
                              return_value=mock_df):
                result = client.get_hourly_outage_capacity(
                    date(2024, 6, 8), date(2024, 6, 8)
                )

        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"
        assert result.index.name == "interval_start_utc"

    def test_expected_column_names(self):
        """All five outage columns are present in the result."""
        client = ERCOTOutageClient()
        mock_df = _make_mock_outage_df()

        with patch.object(client, "_get_now", return_value=_fixed_now()):
            with patch.object(client._backend, "get_hourly_resource_outage_capacity",
                              return_value=mock_df):
                result = client.get_hourly_outage_capacity(
                    date(2024, 6, 8), date(2024, 6, 8)
                )

        assert list(result.columns) == _EXPECTED_COLS

    def test_us_central_timestamps_converted_to_utc(self):
        """US/Central input timestamps are shifted to UTC (+6 h in winter, +5 in summer)."""
        client = ERCOTOutageClient()
        # June is CDT (UTC-5), so 00:00 CDT = 05:00 UTC
        mock_df = _make_mock_outage_df(n_hours=1, start_str="2024-06-08 00:00")

        with patch.object(client, "_get_now", return_value=_fixed_now()):
            with patch.object(client._backend, "get_hourly_resource_outage_capacity",
                              return_value=mock_df):
                result = client.get_hourly_outage_capacity(
                    date(2024, 6, 8), date(2024, 6, 8)
                )

        first_ts = result.index[0]
        assert first_ts.hour == 5  # 00:00 CDT (UTC-5) → 05:00 UTC

    def test_raises_on_historical_date(self):
        """OutageDataError is raised when start is older than 31-day MIS window."""
        client = ERCOTOutageClient()
        old_date = (_fixed_now() - pd.Timedelta(days=_HISTORICAL_COVERAGE_DAYS + 5)).date()

        with patch.object(client, "_get_now", return_value=_fixed_now()):
            with pytest.raises(OutageDataError, match="31-day"):
                client.get_hourly_outage_capacity(old_date, old_date)

    def test_empty_raw_returns_empty_df_with_correct_schema(self):
        """Empty DataFrame from gridstatus is handled gracefully."""
        client = ERCOTOutageClient()
        empty_df = pd.DataFrame()

        result = client._clean_outage(empty_df)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == _EXPECTED_COLS

    def test_aggregate_format_falls_back_gracefully(self):
        """Older gridstatus format (no zone columns) still produces total_outage_mw."""
        client = ERCOTOutageClient()
        agg_df = _make_aggregate_mock_df()

        result = client._clean_outage(agg_df)

        assert "total_outage_mw" in result.columns
        assert result["total_outage_mw"].notna().all()

    def test_retry_on_transient_error_succeeds(self):
        """Client retries on transient errors and returns data on the third attempt."""
        from requests.exceptions import ConnectionError as ReqConnError

        client = ERCOTOutageClient()
        mock_df = _make_mock_outage_df()

        call_count = 0

        def flaky(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ReqConnError("transient error")
            return mock_df

        with patch.object(client, "_get_now", return_value=_fixed_now()):
            with patch.object(
                client._backend, "get_hourly_resource_outage_capacity",
                side_effect=flaky,
            ):
                with patch("energy_trading.data.outages.time.sleep"):
                    result = client.get_hourly_outage_capacity(
                        date(2024, 6, 8), date(2024, 6, 8)
                    )

        assert call_count == 3
        assert len(result) == 24

    def test_raises_outage_data_error_after_max_retries(self):
        """OutageDataError is raised after exhausting all retries."""
        client = ERCOTOutageClient()

        with patch.object(client, "_get_now", return_value=_fixed_now()):
            with patch.object(
                client._backend,
                "get_hourly_resource_outage_capacity",
                side_effect=RuntimeError("always fail"),
            ):
                with patch("energy_trading.data.outages.time.sleep"):
                    with pytest.raises(OutageDataError, match="failed after"):
                        client.get_hourly_outage_capacity(
                            date(2024, 6, 8), date(2024, 6, 8)
                        )
