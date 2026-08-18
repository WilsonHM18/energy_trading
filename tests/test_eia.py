"""Unit tests for energy_trading.data.eia (EIAClient).

All tests use mocked HTTP responses no live network calls.
"""

from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_trading.data.eia import EIAClient, EIAError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_response(data_rows: list[dict], status_code: int = 200) -> MagicMock:
    """Build a fake requests.Response for a given list of row dicts."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        from requests.exceptions import HTTPError
        resp.raise_for_status.side_effect = HTTPError(f"{status_code} error")
    resp.json.return_value = {"response": {"data": data_rows}}
    return resp


def _hourly_rows(
    start: str = "2023-06-01T00",
    n: int = 5,
    value: str = "1000",
    extra: dict | None = None,
) -> list[dict]:
    """Generate synthetic hourly EIA rows."""
    rows = []
    base_date = start[:10]
    base_hour = int(start[11:13])
    for i in range(n):
        h = (base_hour + i) % 24
        day_offset = (base_hour + i) // 24
        # Simple day increment (not DST-aware fine for testing)
        from datetime import timedelta, datetime
        dt = datetime.strptime(base_date, "%Y-%m-%d") + timedelta(days=day_offset)
        row = {"period": f"{dt.date()}T{h:02d}", "value": value}
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def _daily_rows(
    start: str = "2023-06-01",
    n: int = 3,
    value: str = "2.50",
) -> list[dict]:
    from datetime import datetime, timedelta
    rows = []
    base = datetime.strptime(start, "%Y-%m-%d")
    for i in range(n):
        dt = base + timedelta(days=i)
        rows.append({"period": dt.strftime("%Y-%m-%d"), "value": value})
    return rows


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def test_eia_client_raises_on_empty_key():
    with pytest.raises(ValueError, match="API key"):
        EIAClient(api_key="")


def test_eia_client_raises_on_none_key():
    with pytest.raises(ValueError):
        EIAClient(api_key=None)


# ---------------------------------------------------------------------------
# get_wind_generation
# ---------------------------------------------------------------------------


def test_get_wind_generation_returns_dataframe():
    client = EIAClient(api_key="TESTKEY")
    rows = _hourly_rows(n=3, value="12000")
    with patch.object(client._session, "get", return_value=_make_mock_response(rows)):
        result = client.get_wind_generation(date(2023, 6, 1), date(2023, 6, 1))
    assert isinstance(result, pd.DataFrame)
    assert "wind_actual_mw" in result.columns
    assert result.index.name == "interval_start_utc"
    assert len(result) == 3


def test_get_wind_generation_utc_index():
    client = EIAClient(api_key="TESTKEY")
    rows = _hourly_rows(n=2, value="5000")
    with patch.object(client._session, "get", return_value=_make_mock_response(rows)):
        result = client.get_wind_generation(date(2023, 6, 1), date(2023, 6, 1))
    assert result.index.tz is not None
    assert str(result.index.tz) == "UTC"


def test_get_wind_generation_empty_response():
    client = EIAClient(api_key="TESTKEY")
    with patch.object(client._session, "get", return_value=_make_mock_response([])):
        result = client.get_wind_generation(date(2023, 6, 1), date(2023, 6, 1))
    assert result.empty
    assert "wind_actual_mw" in result.columns


# ---------------------------------------------------------------------------
# get_solar_generation
# ---------------------------------------------------------------------------


def test_get_solar_generation_column_name():
    client = EIAClient(api_key="TESTKEY")
    rows = _hourly_rows(n=2, value="3000")
    with patch.object(client._session, "get", return_value=_make_mock_response(rows)):
        result = client.get_solar_generation(date(2023, 6, 1), date(2023, 6, 1))
    assert "solar_actual_mw" in result.columns


# ---------------------------------------------------------------------------
# get_load_forecast
# ---------------------------------------------------------------------------


def test_get_load_forecast_column_name():
    client = EIAClient(api_key="TESTKEY")
    rows = _hourly_rows(n=4, value="50000")
    with patch.object(client._session, "get", return_value=_make_mock_response(rows)):
        result = client.get_load_forecast(date(2023, 6, 1), date(2023, 6, 1))
    assert "load_forecast_mw" in result.columns
    assert len(result) == 4


# ---------------------------------------------------------------------------
# get_gas_price
# ---------------------------------------------------------------------------


def test_get_gas_price_returns_daily_dataframe():
    client = EIAClient(api_key="TESTKEY")
    rows = _daily_rows(n=3, value="2.75")
    with patch.object(client._session, "get", return_value=_make_mock_response(rows)):
        result = client.get_gas_price(date(2023, 6, 1), date(2023, 6, 3))
    assert "gas_price_mmbtu" in result.columns
    assert len(result) == 3


def test_get_gas_price_numeric():
    client = EIAClient(api_key="TESTKEY")
    rows = _daily_rows(n=2, value="3.14")
    with patch.object(client._session, "get", return_value=_make_mock_response(rows)):
        result = client.get_gas_price(date(2023, 6, 1), date(2023, 6, 2))
    assert pd.api.types.is_float_dtype(result["gas_price_mmbtu"])
    import numpy as np
    np.testing.assert_allclose(result["gas_price_mmbtu"].values, 3.14)


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


def test_retry_on_transient_error():
    """Client should retry on network errors and succeed on third attempt."""
    from requests.exceptions import ConnectionError as ReqConnError

    client = EIAClient(api_key="TESTKEY")
    rows = _hourly_rows(n=2, value="9000")
    success_resp = _make_mock_response(rows)

    call_count = 0

    def flaky_get(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ReqConnError("transient")
        return success_resp

    with patch.object(client._session, "get", side_effect=flaky_get):
        with patch("energy_trading.data.eia.time.sleep"):  # speed up test
            result = client.get_wind_generation(date(2023, 6, 1), date(2023, 6, 1))

    assert len(result) == 2
    assert call_count == 3


def test_raises_eia_error_after_max_retries():
    """EIAError should be raised after exhausting all retries."""
    from requests.exceptions import ConnectionError as ReqConnError

    client = EIAClient(api_key="TESTKEY")

    with patch.object(client._session, "get", side_effect=ReqConnError("fail")):
        with patch("energy_trading.data.eia.time.sleep"):
            with pytest.raises(EIAError, match="failed after"):
                client.get_wind_generation(date(2023, 6, 1), date(2023, 6, 1))


def test_eia_error_raised_immediately_on_api_error():
    """An error in the API response body should raise EIAError without retry."""
    client = EIAClient(api_key="TESTKEY")
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"response": {"error": "invalid api_key"}}

    with patch.object(client._session, "get", return_value=resp):
        with pytest.raises(EIAError, match="API error"):
            client.get_wind_generation(date(2023, 6, 1), date(2023, 6, 1))


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def test_pagination_fetches_multiple_pages():
    """Paginator should keep fetching until a page shorter than PAGE_SIZE."""
    from energy_trading.data.eia import _PAGE_SIZE

    client = EIAClient(api_key="TESTKEY")

    page1 = _hourly_rows(start="2023-01-01T00", n=_PAGE_SIZE, value="100")
    page2 = _hourly_rows(start="2023-01-01T00", n=10, value="100")  # final page

    call_count = 0

    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_mock_response(page1)
        return _make_mock_response(page2)

    with patch.object(client._session, "get", side_effect=side_effect):
        result = client.get_wind_generation(date(2023, 1, 1), date(2023, 12, 31))

    assert call_count == 2
    assert len(result) == _PAGE_SIZE + 10
