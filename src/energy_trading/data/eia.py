"""EIA Open Data API client.

Fetches supplementary ERCOT market data from the U.S. Energy Information
Administration (EIA) API v2.  This module covers the features that are not
available from ERCOT's public data portal via gridstatus:

* **Wind generation** hourly actuals for ERCOT (MWh ≡ MW for hourly data)
* **Solar generation** hourly actuals for ERCOT
* **Load forecast** EIA's hourly demand forecast for ERCOT (published in
  near-real-time; used as a proxy for the pre-DAM load signal)
* **Natural gas spot price** daily Henry Hub ($/MMBtu), forward-filled to
  hourly and lagged in the feature layer

EIA API v2 reference
---------------------
* Docs: https://www.eia.gov/opendata/documentation.php
* Key registration: https://www.eia.gov/opendata/
* Electricity RTO endpoint: ``/electricity/rto/fuel-type-data/data/``
* Region-data endpoint: ``/electricity/rto/region-data/data/``
* Natural gas price endpoint: ``/natural-gas/pri/sum/data/``

Timezone note
-------------
EIA electricity/rto data uses **UTC** periods in the format ``"YYYY-MM-DDTHH"``.
Gas prices are daily (format ``"YYYY-MM-DD"``) and are stored with a
midnight-UTC timestamp.

All returned DataFrames use a UTC ``DatetimeIndex`` named
``interval_start_utc``, consistent with the rest of the pipeline.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Any

import pandas as pd
import requests
from loguru import logger

_BASE_URL = "https://api.eia.gov/v2"
_PAGE_SIZE = 5_000          # EIA max rows per request
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0
_COL_TIME = "interval_start_utc"


class EIAError(RuntimeError):
    """Raised when an EIA API call fails after all retries."""


class EIAClient:
    """Client for ERCOT-related data from the EIA Open Data API v2.

    Args:
        api_key: EIA API key.  Obtain one free at
            https://www.eia.gov/opendata/

    Examples:
        >>> client = EIAClient(api_key="YOUR_KEY")
        >>> wind = client.get_wind_generation(date(2022, 1, 1), date(2023, 12, 31))
        >>> wind.head()
    """

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError(
                "EIA API key is required.  Register free at https://www.eia.gov/opendata/"
            )
        self._api_key = api_key
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        logger.debug("EIAClient initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_wind_generation(self, start: date, end: date) -> pd.DataFrame:
        """Fetch hourly ERCOT wind generation actuals.

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).

        Returns:
            DataFrame indexed by ``interval_start_utc`` with column
            ``wind_actual_mw``.
        """
        logger.info("Fetching EIA wind generation: {} → {}", start, end)
        return self._get_fuel_type(start, end, fueltype="WND", col="wind_actual_mw")

    def get_solar_generation(self, start: date, end: date) -> pd.DataFrame:
        """Fetch hourly ERCOT solar (photovoltaic) generation actuals.

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).

        Returns:
            DataFrame indexed by ``interval_start_utc`` with column
            ``solar_actual_mw``.
        """
        logger.info("Fetching EIA solar generation: {} → {}", start, end)
        return self._get_fuel_type(start, end, fueltype="SUN", col="solar_actual_mw")

    def get_load_forecast(self, start: date, end: date) -> pd.DataFrame:
        """Fetch hourly ERCOT demand forecast from EIA.

        This is EIA's own demand forecast for the ERCOT region, published
        in near-real-time.  It is used as a proxy for the pre-DAM load
        expectation signal in feature engineering.

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).

        Returns:
            DataFrame indexed by ``interval_start_utc`` with column
            ``load_forecast_mw``.
        """
        logger.info("Fetching EIA load forecast: {} → {}", start, end)
        return self._get_region_data(start, end, data_type="DF", col="load_forecast_mw")

    def get_gas_price(self, start: date, end: date) -> pd.DataFrame:
        """Fetch daily Henry Hub natural gas spot price ($/MMBtu).

        Henry Hub is the primary gas price benchmark for US power markets.
        ERCOT's marginal heat rate is typically 7–9 MMBtu/MWh, so a
        $1/MMBtu move in gas translates to ~$7–9/MWh in the power spread.

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).

        Returns:
            DataFrame indexed by ``interval_start_utc`` (midnight UTC,
            daily frequency) with column ``gas_price_mmbtu``.
        """
        logger.info("Fetching EIA Henry Hub gas price: {} → {}", start, end)
        return self._get_henry_hub(start, end)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_fuel_type(
        self, start: date, end: date, fueltype: str, col: str
    ) -> pd.DataFrame:
        """Fetch hourly ERCOT fuel-type generation from the EIA RTO endpoint."""
        rows = self._paginate(
            endpoint="electricity/rto/fuel-type-data/data/",
            facets={"respondent": ["ERCO"], "fueltype": [fueltype]},
            start=start,
            end=end,
            frequency="hourly",
        )
        if not rows:
            logger.warning("No EIA fuel-type data returned for fueltype={}", fueltype)
            return pd.DataFrame(columns=[col])

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
        df.index.name = _COL_TIME
        df[col] = pd.to_numeric(df["value"], errors="coerce")
        return df[[col]].sort_index()

    def _get_region_data(
        self, start: date, end: date, data_type: str, col: str
    ) -> pd.DataFrame:
        """Fetch hourly ERCOT region-level data (demand, demand forecast, etc.)."""
        rows = self._paginate(
            endpoint="electricity/rto/region-data/data/",
            facets={"respondent": ["ERCO"], "type": [data_type]},
            start=start,
            end=end,
            frequency="hourly",
        )
        if not rows:
            logger.warning("No EIA region data returned for type={}", data_type)
            return pd.DataFrame(columns=[col])

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
        df.index.name = _COL_TIME
        df[col] = pd.to_numeric(df["value"], errors="coerce")
        return df[[col]].sort_index()

    def _get_henry_hub(self, start: date, end: date) -> pd.DataFrame:
        """Fetch daily Henry Hub spot price from EIA natural-gas/pri/fut endpoint.

        The Henry Hub daily spot price series is ``RNGWHHD`` ($/MMBtu), published
        under the natural-gas futures/spot endpoint.  Weekends and holidays are
        skipped (no trading) forward-fill to hourly is applied in the feature
        layer.
        """
        rows = self._paginate(
            endpoint="natural-gas/pri/fut/data/",
            facets={"series": ["RNGWHHD"]},
            start=start,
            end=end,
            frequency="daily",
        )
        if not rows:
            logger.warning("No EIA Henry Hub gas price data returned.")
            return pd.DataFrame(columns=["gas_price_mmbtu"])

        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["period"], utc=True)
        df.index.name = _COL_TIME
        df["gas_price_mmbtu"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["gas_price_mmbtu"]].sort_index()

    def _paginate(
        self,
        endpoint: str,
        facets: dict[str, list[str]],
        start: date,
        end: date,
        frequency: str,
    ) -> list[dict[str, Any]]:
        """Fetch all pages from an EIA v2 endpoint.

        Handles EIA's offset-based pagination transparently.  Each page
        fetches up to ``_PAGE_SIZE`` rows; iteration continues until fewer
        than ``_PAGE_SIZE`` rows are returned (final page).

        Args:
            endpoint: EIA v2 path (e.g. ``"electricity/rto/fuel-type-data/data/"``).
            facets: Filter criteria (e.g. ``{"respondent": ["ERCO"]}``.
            start: Start date.
            end: End date.
            frequency: ``"hourly"`` or ``"daily"``.

        Returns:
            Flat list of row dicts from all pages combined.
        """
        all_rows: list[dict] = []
        offset = 0

        # Format start/end to match EIA's expected period format.
        if frequency == "hourly":
            start_str = f"{start}T00"
            end_str = f"{end}T23"
        else:
            start_str = str(start)
            end_str = str(end)

        while True:
            params: dict[str, Any] = {
                "api_key": self._api_key,
                "frequency": frequency,
                "data[0]": "value",
                "start": start_str,
                "end": end_str,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "length": _PAGE_SIZE,
                "offset": offset,
            }
            # Facets are passed as repeated query params: facets[key][]=val.
            # Passing a list lets requests encode multiple values correctly.
            for key, values in facets.items():
                params[f"facets[{key}][]"] = values if len(values) > 1 else values[0]

            url = f"{_BASE_URL}/{endpoint}"
            rows = self._get_with_retry(url, params)
            all_rows.extend(rows)

            logger.debug(
                "EIA page offset={}: {} rows (total so far: {})",
                offset,
                len(rows),
                len(all_rows),
            )

            if len(rows) < _PAGE_SIZE:
                break
            offset += _PAGE_SIZE

        return all_rows

    def _get_with_retry(self, url: str, params: dict[str, Any]) -> list[dict]:
        """GET ``url`` with ``params``, retrying on transient errors.

        Args:
            url: Full EIA API URL.
            params: Query parameters dict.

        Returns:
            List of row dicts from ``response["data"]``.

        Raises:
            EIAError: If all retries fail or the API returns an error.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self._session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                payload = resp.json()

                if "response" not in payload:
                    raise EIAError(f"Unexpected EIA response structure: {payload}")

                response_body = payload["response"]

                if "error" in response_body:
                    raise EIAError(f"EIA API error: {response_body['error']}")

                return response_body.get("data", [])

            except EIAError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "EIA request attempt {}/{} failed: {}. Retrying in {:.1f}s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise EIAError(
            f"EIA request failed after {_MAX_RETRIES} attempts"
        ) from last_exc
