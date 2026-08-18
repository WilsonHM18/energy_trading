"""ERCOT generator outage capacity data client.

Historical coverage
-------------------
ERCOT's Hourly Resource Outage Capacity (NP3-233-CD) is published hourly via
the unauthenticated MIS API and retained for only the most recent **31 days**.

For the 2020-2024 historical backtest window, the data is not automatically
accessible without credentials or manual download:

* **Dec 2023 – present**: ERCOT Public API (free account required).
  Register at https://developer.ercot.com/ and use ``gridstatus.ErcotAPI``.
* **2020 – Nov 2023**: Manual bulk download from the ERCOT Data Portal at
  https://data.ercot.com/data-product-archive/NP3-233-CD
  (~1.7 KB per hourly ZIP, ~35 000 files for 2020–2023).

This client is suited for **near-real-time or rolling 30-day production use**.
When outage parquet files are absent, ``DataPipeline.build_features_dataset()``
silently skips outage features (consistent with the EIA/weather graceful-skip
pattern).

NP3-233-CD definition
---------------------
``Total Resource MW`` = total MW of active outages from the Outage Scheduler,
**excluding** IRR (wind/solar), new equipment outages, and mothballed units.
It is therefore a proxy for **thermal generator outages only**.

Zones mirror ERCOT's four load zones:

* North   DFW metro load centre
* South   San Antonio / Corpus Christi
* West    Permian Basin / wind corridor
* Houston Greater Houston area
"""

from __future__ import annotations

import time
from datetime import date

import pandas as pd
from loguru import logger

try:
    from gridstatus import Ercot as _ERCOTBackend
except ImportError as exc:
    raise ImportError(
        "gridstatus is required for outage data ingestion. "
        "Install it with: uv add gridstatus"
    ) from exc

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MAX_RETRIES = 3
_RETRY_BACKOFF_S = 2.0

#: ERCOT MIS document retention window (days).  The IceDocListJsonWS servlet
#: only returns documents published within this rolling window.
_HISTORICAL_COVERAGE_DAYS = 31

_EXPECTED_COLS = [
    "total_outage_mw",
    "outage_mw_north",
    "outage_mw_south",
    "outage_mw_west",
    "outage_mw_houston",
]


class OutageDataError(RuntimeError):
    """Raised when outage data cannot be retrieved or is out of the available window.

    Common causes:

    * The requested date range is older than the 31-day MIS retention window.
    * The ERCOT MIS API returned an unexpected response after all retries.
    """


class ERCOTOutageClient:
    """Client for ERCOT hourly generator outage capacity (NP3-233-CD).

    Wraps ``gridstatus.Ercot.get_hourly_resource_outage_capacity()``, which
    fetches from ERCOT's unauthenticated MIS API (IceDocListJsonWS).  Data is
    available for the **last 31 days only**.

    Returns
    -------
    DataFrame indexed by ``interval_start_utc`` (UTC, hourly) with columns:

    * ``total_outage_mw``   system-wide thermal outage capacity (MW)
    * ``outage_mw_north``   North zone thermal outages (MW)
    * ``outage_mw_south``   South zone thermal outages (MW)
    * ``outage_mw_west``    West zone thermal outages (MW)
    * ``outage_mw_houston`` Houston zone thermal outages (MW)

    Examples
    --------
    >>> from datetime import date, timedelta
    >>> client = ERCOTOutageClient()
    >>> start = date.today() - timedelta(days=7)
    >>> df = client.get_hourly_outage_capacity(start, date.today())
    >>> df.columns.tolist()
    ['total_outage_mw', 'outage_mw_north', 'outage_mw_south', 'outage_mw_west', 'outage_mw_houston']
    """

    def __init__(self) -> None:
        self._backend = _ERCOTBackend()
        logger.debug("ERCOTOutageClient initialised.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_hourly_outage_capacity(
        self,
        start: date,
        end: date,
    ) -> pd.DataFrame:
        """Fetch hourly resource outage capacity for a date range.

        Args:
            start: First date (inclusive).  Must be within the last 31 days
                of the current date.
            end: Last date (inclusive).

        Returns:
            DataFrame with UTC ``DatetimeIndex`` named ``interval_start_utc``
            and columns ``total_outage_mw``, ``outage_mw_{north,south,west,
            houston}``.

        Raises:
            OutageDataError: If ``start`` is older than the 31-day MIS
                retention window, or if the fetch fails after retries.
        """
        now = self._get_now()
        cutoff = (now - pd.Timedelta(days=_HISTORICAL_COVERAGE_DAYS)).normalize()
        start_ts = pd.Timestamp(start, tz="UTC")

        if start_ts < cutoff:
            raise OutageDataError(
                f"Requested start {start} is older than the "
                f"{_HISTORICAL_COVERAGE_DAYS}-day ERCOT MIS document retention "
                f"window (cutoff: {cutoff.date()}). "
                "For historical data, use the ERCOT Public API (Dec 2023+) or "
                "download manually from "
                "https://data.ercot.com/data-product-archive/NP3-233-CD."
            )

        logger.info("Fetching ERCOT outage capacity: {} → {}", start, end)
        raw = self._fetch_with_retry(str(start), str(end))
        return self._clean_outage(raw)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_now(self) -> pd.Timestamp:
        """Return the current UTC timestamp.

        Separated into its own method to allow deterministic mocking in tests.
        """
        return pd.Timestamp.now(tz="UTC")

    def _fetch_with_retry(self, start: str, end: str) -> pd.DataFrame:
        """Call gridstatus with exponential-backoff retry on transient errors.

        Args:
            start: ISO date string for the start of the range.
            end: ISO date string for the end of the range.

        Returns:
            Raw DataFrame from gridstatus.

        Raises:
            OutageDataError: If all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                return self._backend.get_hourly_resource_outage_capacity(
                    date=start, end=end
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = _RETRY_BACKOFF_S * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt {}/{} failed for get_hourly_resource_outage_capacity: "
                    "{}. Retrying in {:.1f}s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)

        raise OutageDataError(
            f"Outage data fetch failed after {_MAX_RETRIES} attempts"
        ) from last_exc

    def _clean_outage(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Standardise a gridstatus outage DataFrame to the project schema.

        Handles two gridstatus output formats:

        * **Zone format** (newer): zone-specific columns
          ``Total Resource MW Zone {North,South,West,Houston}`` plus a
          pre-computed ``Total Resource MW`` aggregate.
        * **Aggregate format** (older): only ``Total Resource MW`` present.

        Timestamps are normalised to a UTC ``DatetimeIndex`` named
        ``interval_start_utc``.  Gridstatus returns US/Central-aware
        timestamps; conversion to UTC is applied unconditionally.

        Args:
            raw: DataFrame returned by
                ``gridstatus.Ercot.get_hourly_resource_outage_capacity()``.

        Returns:
            Cleaned DataFrame with columns ``total_outage_mw``,
            ``outage_mw_{north,south,west,houston}``.
        """
        if raw is None or (hasattr(raw, "empty") and raw.empty):
            return pd.DataFrame(columns=_EXPECTED_COLS)

        df = raw.copy()

        # --- Resolve the timestamp index ---
        # gridstatus may return Interval Start as a column or as the index.
        if "Interval Start" in df.columns:
            idx = pd.to_datetime(df["Interval Start"])
        else:
            idx = pd.to_datetime(df.index)

        if idx.tz is None:
            idx = idx.tz_localize("US/Central", ambiguous="infer")
        idx = idx.tz_convert("UTC")
        df.index = idx
        df.index.name = "interval_start_utc"

        # --- Map zone columns to project schema ---
        _zone_map = {
            "Total Resource MW Zone North": "outage_mw_north",
            "Total Resource MW Zone South": "outage_mw_south",
            "Total Resource MW Zone West": "outage_mw_west",
            "Total Resource MW Zone Houston": "outage_mw_houston",
            "Total Resource MW": "total_outage_mw",
        }

        result = pd.DataFrame(index=df.index)
        for src_col, dst_col in _zone_map.items():
            if src_col in df.columns:
                result[dst_col] = pd.to_numeric(df[src_col], errors="coerce")

        # --- Compute total from zones if the direct total column is absent ---
        if "total_outage_mw" not in result.columns:
            zone_cols = [c for c in result.columns if c.startswith("outage_mw_")]
            if zone_cols:
                result["total_outage_mw"] = result[zone_cols].sum(axis=1)
            else:
                result["total_outage_mw"] = float("nan")

        # --- Ensure all expected output columns exist (fill absent with NaN) ---
        for col in _EXPECTED_COLS:
            if col not in result.columns:
                result[col] = float("nan")

        return result[_EXPECTED_COLS].sort_index()
