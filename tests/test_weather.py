"""Unit tests for energy_trading.data.weather (WeatherClient).

All tests use mocked HTTP responses -- no live network calls.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_trading.data.weather import WeatherClient, WeatherError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(times: list[str], temps: list[float]) -> MagicMock:
    """Build a fake requests.Response for Open-Meteo data."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {
        "latitude": 30.267,
        "longitude": -97.743,
        "timezone": "GMT",
        "hourly": {
            "time": times,
            "temperature_2m": temps,
        },
    }
    return resp


def _make_hourly_times(start: str, n_hours: int) -> list[str]:
    """Generate Open-Meteo-style ISO8601 UTC time strings."""
    idx = pd.date_range(start=start, periods=n_hours, freq="h", tz="UTC")
    return [ts.strftime("%Y-%m-%dT%H:%M") for ts in idx]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWeatherClient:
    def test_get_temperature_returns_dataframe(self):
        client = WeatherClient()
        times = _make_hourly_times("2023-01-01T00:00", 24)
        temps = [10.0 + i * 0.5 for i in range(24)]
        with patch.object(client._session, "get", return_value=_make_mock_response(times, temps)):
            result = client.get_temperature(30.267, -97.743, date(2023, 1, 1), date(2023, 1, 1))
        assert isinstance(result, pd.DataFrame)

    def test_get_temperature_index_is_utc_datetimeindex(self):
        client = WeatherClient()
        times = _make_hourly_times("2023-01-01T00:00", 24)
        temps = [15.0] * 24
        with patch.object(client._session, "get", return_value=_make_mock_response(times, temps)):
            result = client.get_temperature(30.267, -97.743, date(2023, 1, 1), date(2023, 1, 1))
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"
        assert result.index.name == "interval_start_utc"

    def test_get_temperature_column_name(self):
        client = WeatherClient()
        times = _make_hourly_times("2023-01-01T00:00", 24)
        temps = [20.0] * 24
        with patch.object(client._session, "get", return_value=_make_mock_response(times, temps)):
            result = client.get_temperature(30.267, -97.743, date(2023, 1, 1), date(2023, 1, 1))
        assert "temperature_c" in result.columns

    def test_get_temperature_row_count(self):
        """Row count should equal (end - start).days * 24 + 24 (inclusive both ends)."""
        client = WeatherClient()
        n_hours = 48  # 2 days
        times = _make_hourly_times("2023-01-01T00:00", n_hours)
        temps = [10.0] * n_hours
        with patch.object(client._session, "get", return_value=_make_mock_response(times, temps)):
            result = client.get_temperature(30.267, -97.743, date(2023, 1, 1), date(2023, 1, 2))
        assert len(result) == n_hours

    def test_get_temperature_values_numeric(self):
        """Temperature values should be float64 with plausible Texas range."""
        client = WeatherClient()
        times = _make_hourly_times("2023-07-01T00:00", 24)
        temps = [35.0 + i * 0.1 for i in range(24)]  # realistic summer temps
        with patch.object(client._session, "get", return_value=_make_mock_response(times, temps)):
            result = client.get_temperature(30.267, -97.743, date(2023, 7, 1), date(2023, 7, 1))
        assert pd.api.types.is_float_dtype(result["temperature_c"])
        assert result["temperature_c"].notna().all()
        assert (result["temperature_c"] > -50).all()  # plausible range
        assert (result["temperature_c"] < 60).all()

    def test_empty_response_returns_empty_df(self):
        """Empty hourly data from API should return empty DataFrame with correct schema."""
        client = WeatherClient()
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"hourly": {"time": [], "temperature_2m": []}}
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_temperature(30.267, -97.743, date(2023, 1, 1), date(2023, 1, 1))
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert "temperature_c" in result.columns

    def test_retry_on_transient_error(self):
        """Client should retry on network errors and succeed on third attempt."""
        from requests.exceptions import ConnectionError as ReqConnError

        client = WeatherClient()
        times = _make_hourly_times("2023-01-01T00:00", 24)
        temps = [15.0] * 24
        success_resp = _make_mock_response(times, temps)

        call_count = 0

        def flaky_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ReqConnError("transient error")
            return success_resp

        with patch.object(client._session, "get", side_effect=flaky_get):
            with patch("energy_trading.data.weather.time.sleep"):  # speed up
                result = client.get_temperature(30.267, -97.743, date(2023, 1, 1), date(2023, 1, 1))

        assert call_count == 3
        assert len(result) == 24

    def test_raises_weather_error_after_max_retries(self):
        """WeatherError should be raised after exhausting all retries."""
        from requests.exceptions import ConnectionError as ReqConnError

        client = WeatherClient()

        with patch.object(client._session, "get", side_effect=ReqConnError("always fail")):
            with patch("energy_trading.data.weather.time.sleep"):
                with pytest.raises(WeatherError, match="failed after"):
                    client._fetch_openmeteo({"latitude": 30.267, "longitude": -97.743})

    def test_raises_on_http_error(self):
        """HTTP 4xx/5xx should raise WeatherError after retries."""
        from requests.exceptions import HTTPError

        client = WeatherClient()
        resp = MagicMock()
        resp.raise_for_status.side_effect = HTTPError("429 Too Many Requests")

        with patch.object(client._session, "get", return_value=resp):
            with patch("energy_trading.data.weather.time.sleep"):
                with pytest.raises(WeatherError, match="failed after"):
                    client._fetch_openmeteo({"latitude": 30.267, "longitude": -97.743})

    def test_sorted_ascending(self):
        """Result index should be monotonically increasing."""
        client = WeatherClient()
        # Deliberately provide times out of order
        times = _make_hourly_times("2023-06-01T00:00", 24)
        times_reversed = list(reversed(times))
        temps = list(range(24))
        with patch.object(
            client._session, "get",
            return_value=_make_mock_response(times_reversed, temps)
        ):
            result = client.get_temperature(30.267, -97.743, date(2023, 6, 1), date(2023, 6, 1))
        assert result.index.is_monotonic_increasing
