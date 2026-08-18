"""ERCOT market data client.

Wraps the ``gridstatus`` library to fetch historical ERCOT settlement-point
prices and actual load.  Due to how ERCOT's public data portal is structured,
prices and load are downloaded as **full-year bulk files**; the ``years``
parameter controls which calendar years are fetched.

Returned DataFrames all share a common schema:
* Index: ``pandas.DatetimeIndex`` named ``interval_start_utc`` in UTC.
* Prices are in **$/MWh**.  Load is in **MW**.

ERCOT market timing
-------------------
* DAM closes ~10:00 AM CPT the day before delivery.
* Real-Time (SCED) runs every 5 min; 15-min RTM prices are published shortly
  after each quarter-hour interval.
* RTM prices are averaged to hourly before storage.

References
----------
* ERCOT Nodal Protocols: https://www.ercot.com/mktrules/nprotocols
* gridstatus docs: https://docs.gridstatus.io
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from loguru import logger

try:
    from gridstatus import Ercot as _ERCOTBackend
except ImportError as exc:
    raise ImportError(
        "gridstatus is required for data ingestion. "
        "Install it with: uv add gridstatus"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_HUBS: list[str] = ["HB_NORTH", "HB_SOUTH", "HB_WEST", "HB_HOUSTON"]
_LOCATION_TYPE_HUB = "Trading Hub"
_COL_TIME = "interval_start_utc"
_COL_HUB = "hub"
_COL_LMP = "lmp"

_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0


class ERCOTDataError(RuntimeError):
    """Raised when data cannot be retrieved or is structurally invalid."""


class ERCOTClient:
    """Client for ERCOT historical price and load data via ``gridstatus``.

    All price and load data are fetched as full-year bulk files from ERCOT's
    public data portal.  Partial-year windows are applied at the pipeline
    level after download.

    Args:
        hubs: Settlement-point trading hubs to include.  Defaults to the
            four main ERCOT hubs.

    Examples:
        >>> client = ERCOTClient()
        >>> dam = client.get_dam_prices([2022, 2023])
        >>> dam.head()
    """

    def __init__(self, hubs: list[str] | None = None) -> None:
        self._hubs = [h.upper() for h in (hubs or DEFAULT_HUBS)]
        self._backend = _ERCOTBackend()
        logger.debug("ERCOTClient initialised with hubs={}", self._hubs)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dam_prices(
        self,
        years: list[int],
        hubs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch hourly Day-Ahead Market settlement-point prices.

        Downloads full-year bulk files from ERCOT for each requested year
        and returns a single concatenated DataFrame filtered to the target
        hubs.

        Args:
            years: Calendar years to fetch (e.g. ``[2021, 2022, 2023]``).
            hubs: Hub names to retain.  Defaults to ``self._hubs``.

        Returns:
            DataFrame indexed by ``interval_start_utc`` with columns
            ``hub`` and ``lmp``.
        """
        target_hubs = [h.upper() for h in (hubs or self._hubs)]
        logger.info("Fetching DAM prices for years={}, hubs={}", years, target_hubs)
        frames = [
            self._fetch_with_retry(self._backend.get_dam_spp, year=yr)
            for yr in sorted(years)
        ]
        return self._clean_spp(pd.concat(frames, ignore_index=True), target_hubs)

    def get_rtm_prices(
        self,
        years: list[int],
        hubs: list[str] | None = None,
    ) -> pd.DataFrame:
        """Fetch Real-Time Market prices, resampled to hourly averages.

        Downloads full-year bulk files (15-minute granularity) and resamples
        to hourly by simple arithmetic mean of the four 15-minute intervals.
        This matches ERCOT's virtual-bid settlement convention.

        Args:
            years: Calendar years to fetch.
            hubs: Hub names to retain.  Defaults to ``self._hubs``.

        Returns:
            Hourly DataFrame with the same schema as ``get_dam_prices``.
        """
        target_hubs = [h.upper() for h in (hubs or self._hubs)]
        logger.info("Fetching RTM prices for years={}, hubs={}", years, target_hubs)
        frames = [
            self._fetch_with_retry(self._backend.get_rtm_spp, year=yr)
            for yr in sorted(years)
        ]
        df_15min = self._clean_spp(pd.concat(frames, ignore_index=True), target_hubs)
        return self._resample_to_hourly(df_15min)

    def get_load_actual(self, years: list[int]) -> pd.DataFrame:
        """Fetch ERCOT-wide actual system load at hourly resolution.

        Calls ``get_hourly_load_post_settlements`` for each year.  The source
        data includes load by weather zone; this method returns the system-wide
        total (``ERCOT`` column) only.

        Args:
            years: Calendar years to fetch.

        Returns:
            DataFrame indexed by ``interval_start_utc`` with column
            ``load_mw``.
        """
        logger.info("Fetching actual load for years={}", years)
        frames = []
        for yr in sorted(years):
            # gridstatus fetches the full year regardless of date arguments;
            # we pass the first day of each year as the anchor.
            raw = self._fetch_with_retry(
                self._backend.get_hourly_load_post_settlements,
                date=f"{yr}-01-01",
                end=f"{yr}-12-31",
            )
            frames.append(raw)
        return self._clean_load(pd.concat(frames, ignore_index=True))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_with_retry(self, fn: Any, **kwargs: Any) -> pd.DataFrame:
        """Call ``fn(**kwargs)`` with exponential-backoff retry on failure.

        Args:
            fn: Callable that returns a DataFrame.
            **kwargs: Arguments forwarded to ``fn``.

        Returns:
            DataFrame result from ``fn``.

        Raises:
            ERCOTDataError: If all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = fn(**kwargs)
                if not isinstance(result, pd.DataFrame):
                    raise ERCOTDataError(
                        f"{fn.__name__} returned {type(result)!r}, expected DataFrame"
                    )
                return result
            except ERCOTDataError:
                raise
            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt {}/{} failed for {}: {}. Retrying in {:.1f}s",
                    attempt,
                    _MAX_RETRIES,
                    fn.__name__,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise ERCOTDataError(
            f"Failed to fetch data after {_MAX_RETRIES} attempts"
        ) from last_exc

    def _clean_spp(self, raw: pd.DataFrame, target_hubs: list[str]) -> pd.DataFrame:
        """Standardise a raw gridstatus SPP DataFrame.

        Normalises column names, converts the interval-start timestamp to a
        UTC DatetimeIndex, and filters to the requested trading hubs.

        The raw DataFrame from ``get_dam_spp`` / ``get_rtm_spp`` has columns::

            Time, Interval Start, Interval End, Location, Location Type,
            Market, SPP

        Args:
            raw: Raw DataFrame from gridstatus.
            target_hubs: Hub names to retain (upper-cased).

        Returns:
            Cleaned DataFrame indexed by ``interval_start_utc`` with columns
            ``hub`` and ``lmp``.
        """
        df = raw.copy()

        # --- Parse and convert the interval-start timestamp to UTC ---
        interval_start = pd.to_datetime(df["Interval Start"])
        if interval_start.dt.tz is None:
            interval_start = interval_start.dt.tz_localize("US/Central", ambiguous="infer")
        df.index = interval_start.dt.tz_convert("UTC")
        df.index.name = _COL_TIME

        # --- Normalise hub column ---
        df[_COL_HUB] = df["Location"].str.upper()

        # --- Filter to Trading Hub location type and target hubs ---
        mask = (df["Location Type"] == _LOCATION_TYPE_HUB) & (df[_COL_HUB].isin(target_hubs))
        df = df.loc[mask].copy()

        if df.empty:
            available = raw["Location"].unique().tolist()
            logger.warning(
                "No rows after filtering to hubs={}.  Available locations: {}",
                target_hubs,
                available[:20],
            )

        # --- Rename SPP to lmp and select output columns ---
        df[_COL_LMP] = df["SPP"]
        return df[[_COL_HUB, _COL_LMP]].sort_index()

    def _resample_to_hourly(self, df_15min: pd.DataFrame) -> pd.DataFrame:
        """Resample 15-minute LMP data to hourly by simple arithmetic mean.

        Groups on (hour-truncated timestamp, hub) and averages the four
        15-minute observations within each clock hour.

        Args:
            df_15min: 15-minute LMP DataFrame with a UTC DatetimeIndex.

        Returns:
            Hourly DataFrame with the same schema.
        """
        hourly = (
            df_15min.groupby([pd.Grouper(freq="h"), _COL_HUB])[[_COL_LMP]]
            .mean()
            .reset_index(level=_COL_HUB)
        )
        return hourly.sort_index()

    def _clean_load(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Standardise a raw load post-settlements DataFrame.

        The source has one row per hour with load broken out by weather zone.
        We retain only the system-wide ``ERCOT`` column and index by UTC.

        Args:
            raw: Raw DataFrame from ``get_hourly_load_post_settlements``.

        Returns:
            DataFrame indexed by ``interval_start_utc`` with column
            ``load_mw``.
        """
        df = raw.copy()

        interval_start = pd.to_datetime(df["Interval Start"])
        if interval_start.dt.tz is None:
            interval_start = interval_start.dt.tz_localize("US/Central", ambiguous="infer")
        df.index = interval_start.dt.tz_convert("UTC")
        df.index.name = _COL_TIME

        if "ERCOT" not in df.columns:
            raise ERCOTDataError(
                f"Expected 'ERCOT' column in load data. Got: {list(df.columns)}"
            )

        return df[["ERCOT"]].rename(columns={"ERCOT": "load_mw"}).sort_index()
