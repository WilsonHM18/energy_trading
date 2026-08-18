"""Open-Meteo weather data client.

Fetches hourly temperature from the Open-Meteo Historical Weather API
(ERA5 reanalysis archive).  The API is free and requires no API key.

Usage
-----
::

    from datetime import date
    from energy_trading.data.weather import WeatherClient

    client = WeatherClient()
    df = client.get_temperature(lat=30.267, lon=-97.743,
                                start=date(2020, 1, 1), end=date(2024, 12, 31))
    # Returns DataFrame with UTC DatetimeIndex and column 'temperature_c'.

API reference
-------------
* Docs: https://open-meteo.com/en/docs/historical-weather-api
* Endpoint: ``https://archive-api.open-meteo.com/v1/archive``
* No API key required.
* Timezone: pass ``timezone=UTC`` to receive all timestamps in UTC.
* Variable: ``temperature_2m`` (°C at 2 m above ground).

Timezone note
-------------
All returned DataFrames use a UTC ``DatetimeIndex`` named
``interval_start_utc``, consistent with the rest of the pipeline.
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests
from loguru import logger

_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0
_COL_TIME = "interval_start_utc"


class WeatherError(RuntimeError):
    """Raised when an Open-Meteo API call fails after all retries."""


class WeatherClient:
    """Client for hourly temperature data from the Open-Meteo archive API.

    No API key is required.  Temperature data comes from the ERA5 reanalysis
    and is available from 1940 to ~5 days before today.

    Examples:
        >>> client = WeatherClient()
        >>> df = client.get_temperature(
        ...     lat=30.267, lon=-97.743,
        ...     start=date(2022, 1, 1), end=date(2022, 12, 31)
        ... )
        >>> df.head()
    """

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers["Accept"] = "application/json"
        logger.debug("WeatherClient initialised (no API key required).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_temperature(
        self,
        lat: float,
        lon: float,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch hourly temperature at 2 m above ground (ERA5 reanalysis).

        Args:
            lat: Latitude of the location (decimal degrees).
            lon: Longitude of the location (decimal degrees).
            start: First date (inclusive).
            end: Last date (inclusive).

        Returns:
            DataFrame indexed by ``interval_start_utc`` (UTC DatetimeIndex,
            hourly) with column ``temperature_c`` (float64, °C).
            Returns an empty DataFrame with the correct schema on failure.
        """
        logger.info(
            "Fetching Open-Meteo temperature: lat={}, lon={}, {} -> {}",
            lat,
            lon,
            start,
            end,
        )
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": str(start),
            "end_date": str(end),
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }

        try:
            data = self._fetch_openmeteo(params)
        except WeatherError:
            logger.warning(
                "Open-Meteo request failed; returning empty temperature DataFrame."
            )
            return pd.DataFrame(columns=["temperature_c"])

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])

        if not times:
            logger.warning("Open-Meteo returned empty hourly data.")
            return pd.DataFrame(columns=["temperature_c"])

        idx = pd.to_datetime(times, utc=True)
        idx.name = _COL_TIME

        df = pd.DataFrame(
            {"temperature_c": pd.array(temps, dtype="float64")},
            index=idx,
        )

        logger.info(
            "Open-Meteo: {} hourly temperature records fetched ({} -> {}).",
            len(df),
            start,
            end,
        )
        return df.sort_index()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_openmeteo(self, params: dict) -> dict:
        """GET the Open-Meteo archive endpoint with exponential-backoff retry.

        Args:
            params: Query parameters dict passed to the API.

        Returns:
            Parsed JSON response dict from Open-Meteo.

        Raises:
            WeatherError: If all retries fail or the response is invalid.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._session.get(_BASE_URL, params=params, timeout=30)
                resp.raise_for_status()
                return resp.json()

            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "Open-Meteo request attempt {}/{} failed: {}. Retrying in {:.1f}s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise WeatherError(
            f"Open-Meteo request failed after {_MAX_RETRIES} attempts"
        ) from last_exc
