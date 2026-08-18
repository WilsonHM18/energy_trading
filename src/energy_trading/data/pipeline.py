"""End-to-end data pipeline for ERCOT market data.

The ``DataPipeline`` class is the single entry point for downloading,
validating, and persisting all datasets required by the modelling phase.

Storage layout
--------------
* ``data/raw/dam_prices_{year}.parquet`` full-year DAM prices per hub
* ``data/raw/rtm_prices_{year}.parquet`` full-year RTM hourly prices per hub
* ``data/raw/load_actual_{year}.parquet`` full-year system-wide load
* ``data/raw/eia_wind_{year}.parquet`` hourly ERCOT wind generation (EIA)
* ``data/raw/eia_solar_{year}.parquet`` hourly ERCOT solar generation (EIA)
* ``data/raw/eia_load_forecast_{year}.parquet`` hourly DA demand forecast (EIA)
* ``data/raw/eia_gas_price_{year}.parquet`` daily Henry Hub spot price (EIA)
* ``data/raw/weather_temperature_{year}.parquet`` hourly temperature (Open-Meteo)
* ``data/raw/outage_capacity_{year}.parquet`` hourly thermal outage capacity (ERCOT)
* ``data/processed/spread_dataset.parquet`` aligned DA/RT spread dataset
* ``data/processed/features_dataset.parquet`` full feature matrix

Idempotency
-----------
Each per-year Parquet file is only fetched once.  Subsequent ``run()`` calls
skip years that already have cached files.  Pass ``force=True`` to
re-download everything.

Spread definition
-----------------
::

    spread_t = dam_lmp_t − rtm_lmp_t   [$/MWh, per hub per hour]

Positive spread → DA > RT → virtual offer (seller) profits.
Negative spread → RT > DA → virtual bid (buyer) profits.

Output schema (``data/processed/spread_dataset.parquet``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Index: ``interval_start_utc`` (UTC, hourly)
Columns:
  * ``hub``                  – ERCOT trading hub
  * ``dam_lmp``              – Day-Ahead LMP ($/MWh)
  * ``rtm_lmp``              – Real-Time LMP, hourly average ($/MWh)
  * ``spread``               – ``dam_lmp − rtm_lmp`` ($/MWh)
  * ``load_mw``              – Actual system-wide load (MW)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from energy_trading.config import Settings, settings as default_settings
from energy_trading.data.eia import EIAClient
from energy_trading.data.ercot import ERCOTClient
from energy_trading.data.validation import DataValidationError, validate_lmp_dataframe
from energy_trading.data.weather import WeatherClient

# Default Open-Meteo location: Austin, TX (central ERCOT)
_WEATHER_LAT = 30.267
_WEATHER_LON = -97.743

_PARQUET_ENGINE = "pyarrow"
_PARQUET_COMPRESSION = "snappy"


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine=_PARQUET_ENGINE, compression=_PARQUET_COMPRESSION)
    logger.debug("Saved {} rows → {}", len(df), path)


def _load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path, engine=_PARQUET_ENGINE)
    logger.debug("Loaded {} rows ← {}", len(df), path)
    return df


class DataPipeline:
    """Orchestrates ERCOT data collection, validation, and storage.

    Args:
        client: ``ERCOTClient`` instance.  Created automatically if omitted.
        cfg: ``Settings`` instance.  Defaults to the module-level singleton.

    Examples:
        >>> from datetime import date
        >>> pipeline = DataPipeline()
        >>> pipeline.run(start=date(2021, 1, 1), end=date(2023, 12, 31))
        >>> df = pipeline.load_spread_dataset()
    """

    def __init__(
        self,
        client: ERCOTClient | None = None,
        cfg: Settings | None = None,
    ) -> None:
        self._cfg = cfg or default_settings
        self._client = client or ERCOTClient(hubs=self._cfg.hubs)
        self._cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        self._cfg.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, start: date, end: date, *, force: bool = False) -> None:
        """Download, validate, and persist all required datasets.

        Fetches data for all calendar years that overlap [start, end].
        Per-year Parquet files that already exist are skipped unless
        ``force=True``.

        Args:
            start: First delivery date (inclusive).
            end: Last delivery date (inclusive).
            force: Re-download and overwrite existing files if ``True``.
        """
        years = list(range(start.year, end.year + 1))
        logger.info(
            "Pipeline run: {} → {} (years={}, force={})", start, end, years, force
        )

        dam = self._load_or_fetch_years("dam_prices", years, force=force)
        rtm = self._load_or_fetch_years("rtm_prices", years, force=force)
        load = self._load_or_fetch_years("load_actual", years, force=force)

        # Trim to the exact requested window before building the spread dataset.
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)

        dam_window = dam.loc[start_ts:end_ts]
        rtm_window = rtm.loc[start_ts:end_ts]
        load_window = load.loc[start_ts:end_ts]

        spread_path = self._cfg.processed_dir / "spread_dataset.parquet"
        if not force and spread_path.exists():
            existing = _load_parquet(spread_path)
            if existing.index.min() <= start_ts and existing.index.max() >= end_ts:
                logger.info("Spread dataset already covers requested range skipping.")
                return
            logger.info(
                "Spread dataset covers {} → {} but {} → {} requested; rebuilding.",
                existing.index.min().date(),
                existing.index.max().date(),
                start,
                end,
            )

        logger.info("Constructing spread dataset…")
        spread_df = self._build_spread_dataset(
            dam=dam_window,
            rtm=rtm_window,
            load=load_window,
        )
        _save_parquet(spread_df, spread_path)
        logger.info(
            "Spread dataset saved: {} rows × {} cols → {}",
            len(spread_df),
            spread_df.shape[1],
            spread_path,
        )

    def load_spread_dataset(self) -> pd.DataFrame:
        """Load the pre-built spread dataset from disk.

        Returns:
            Spread DataFrame as described in the module docstring.

        Raises:
            FileNotFoundError: If ``pipeline.run()`` has not been called yet.
        """
        path = self._cfg.processed_dir / "spread_dataset.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Spread dataset not found at {path}.  Run DataPipeline.run() first."
            )
        return _load_parquet(path)

    def run_eia(self, start: date, end: date, *, force: bool = False) -> None:
        """Download and cache EIA supplementary datasets for the given date range.

        Fetches four datasets per calendar year: wind generation, solar
        generation, day-ahead load forecast, and Henry Hub natural gas spot
        price.  Each dataset is cached per year as a Parquet file under
        ``data/raw/eia_{name}_{year}.parquet``.

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).
            force: Re-download and overwrite existing files if ``True``.

        Raises:
            ValueError: If ``ET_EIA_API_KEY`` is not configured.
        """
        if not self._cfg.eia_api_key:
            raise ValueError(
                "EIA API key not configured.  Set ET_EIA_API_KEY in your .env file."
            )
        eia = EIAClient(api_key=self._cfg.eia_api_key)
        years = list(range(start.year, end.year + 1))
        logger.info("EIA pipeline run: {} → {} (years={})", start, end, years)

        for name, method in [
            ("eia_wind", eia.get_wind_generation),
            ("eia_solar", eia.get_solar_generation),
            ("eia_load_forecast", eia.get_load_forecast),
            ("eia_gas_price", eia.get_gas_price),
        ]:
            self._load_or_fetch_eia_years(name, years, method, force=force)

        logger.info("EIA pipeline run complete.")

    def run_outages(
        self,
        start: date,
        end: date,
        *,
        force: bool = False,
    ) -> None:
        """Download and cache ERCOT hourly outage capacity for the given range.

        Fetches NP3-233-CD (Hourly Resource Outage Capacity) via the
        unauthenticated ERCOT MIS API.  **Coverage is limited to the last
        31 days** earlier dates are silently skipped with a warning.

        For historical 2020–2024 data, the ERCOT Public API (free account,
        Dec 2023+ coverage) or manual bulk download is required; see
        ``ERCOTOutageClient`` for details.

        Each year within the available window is cached as
        ``data/raw/outage_capacity_{year}.parquet``.

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).
            force: Re-download and overwrite existing files if ``True``.
        """
        from energy_trading.data.outages import ERCOTOutageClient, OutageDataError

        cutoff = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=31)).date()

        if end < cutoff:
            logger.warning(
                "run_outages: end date {} is older than the ERCOT MIS 31-day "
                "retention window (cutoff: {}). No outage data downloaded. "
                "For historical data: "
                "https://data.ercot.com/data-product-archive/NP3-233-CD",
                end,
                cutoff,
            )
            return

        effective_start = max(start, cutoff)
        if effective_start > start:
            logger.warning(
                "run_outages: clamping start from {} to {} (31-day MIS limit).",
                start,
                effective_start,
            )

        client = ERCOTOutageClient()
        years = list(range(effective_start.year, end.year + 1))
        logger.info(
            "Outage pipeline run: {} → {} (years={})", effective_start, end, years
        )

        for yr in sorted(years):
            path = self._cfg.raw_dir / f"outage_capacity_{yr}.parquet"
            if not force and path.exists():
                logger.info("Cache hit loading outage_capacity {} from disk.", yr)
                continue

            yr_start = max(effective_start, date(yr, 1, 1))
            yr_end = min(end, date(yr, 12, 31))
            logger.info(
                "Fetching outage_capacity for year={} ({} → {})", yr, yr_start, yr_end
            )
            try:
                yr_df = client.get_hourly_outage_capacity(yr_start, yr_end)
                if not yr_df.empty:
                    _save_parquet(yr_df, path)
                else:
                    logger.warning("No outage data returned for year {}.", yr)
            except OutageDataError as exc:
                logger.warning("Outage data unavailable for {}: {}", yr, exc)

        logger.info("Outage pipeline run complete.")

    def run_weather(
        self,
        start: date,
        end: date,
        *,
        force: bool = False,
        lat: float = _WEATHER_LAT,
        lon: float = _WEATHER_LON,
    ) -> None:
        """Download and cache hourly temperature data from Open-Meteo.

        Fetches ERA5 reanalysis temperature at ``lat``/``lon`` for each
        calendar year in [start, end].  Each year is cached as a Parquet
        file at ``data/raw/weather_temperature_{year}.parquet``.

        No API key is required.

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).
            force: Re-download and overwrite existing files if ``True``.
            lat: Latitude of the weather station (default: Austin TX).
            lon: Longitude of the weather station (default: Austin TX).
        """
        client = WeatherClient()
        years = list(range(start.year, end.year + 1))
        logger.info(
            "Weather pipeline run: {} -> {} (years={}, lat={}, lon={})",
            start,
            end,
            years,
            lat,
            lon,
        )

        for yr in sorted(years):
            path = self._cfg.raw_dir / f"weather_temperature_{yr}.parquet"
            if not force and path.exists():
                logger.info("Cache hit loading weather_temperature {} from disk.", yr)
                continue
            logger.info("Fetching weather_temperature for year={} from Open-Meteo...", yr)
            yr_df = client.get_temperature(lat, lon, date(yr, 1, 1), date(yr, 12, 31))
            if not yr_df.empty:
                _save_parquet(yr_df, path)
            else:
                logger.warning("No weather data returned for year {}.", yr)

        logger.info("Weather pipeline run complete.")

    def build_features_dataset(
        self, start: date, end: date, *, force: bool = False
    ) -> None:
        """Build and save the full feature matrix.

        Loads the spread dataset and any available EIA datasets, runs
        ``build_features()`` and ``add_eia_features()``, and writes the
        result to ``data/processed/features_dataset.parquet``.

        This method is idempotent if the feature dataset already exists
        and covers the requested range, it is skipped (unless ``force=True``).

        Args:
            start: First date (inclusive).
            end: Last date (inclusive).
            force: Rebuild even if the features dataset already exists.

        Raises:
            FileNotFoundError: If ``pipeline.run()`` has not been called yet.
        """
        from energy_trading.features.engineering import (
            add_eia_features,
            add_outage_features,
            add_weather_features,
            build_features,
        )

        out_path = self._cfg.processed_dir / "features_dataset.parquet"
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)

        if not force and out_path.exists():
            existing = _load_parquet(out_path)
            if existing.index.min() <= start_ts and existing.index.max() >= end_ts:
                logger.info("Features dataset already covers requested range skipping.")
                return
            logger.info("Features dataset does not cover requested range; rebuilding.")

        spread_df = self.load_spread_dataset().loc[start_ts:end_ts]
        logger.info("Building feature matrix for {} rows…", len(spread_df))

        features_df = build_features(spread_df)

        # Optionally merge EIA features if cached files exist.
        years = list(range(start.year, end.year + 1))
        wind = self._try_load_eia("eia_wind", years, start_ts, end_ts)
        solar = self._try_load_eia("eia_solar", years, start_ts, end_ts)
        load_fc = self._try_load_eia("eia_load_forecast", years, start_ts, end_ts)
        gas = self._try_load_eia("eia_gas_price", years, start_ts, end_ts)

        features_df = add_eia_features(
            features_df,
            wind_actual=wind,
            solar_actual=solar,
            load_forecast=load_fc,
            gas_price=gas,
        )

        # Optionally merge weather features if cached files exist.
        temperature = self._try_load_weather(years, start_ts, end_ts)
        features_df = add_weather_features(features_df, temperature=temperature)

        # Optionally merge outage features if cached files exist.
        outage_capacity = self._try_load_outages(years, start_ts, end_ts)
        features_df = add_outage_features(features_df, outage_capacity=outage_capacity)

        _save_parquet(features_df, out_path)
        logger.info(
            "Features dataset saved: {} rows × {} cols → {}",
            len(features_df),
            features_df.shape[1],
            out_path,
        )

    def load_features_dataset(self) -> pd.DataFrame:
        """Load the pre-built feature matrix from disk.

        Returns:
            Feature DataFrame produced by ``build_features_dataset()``.

        Raises:
            FileNotFoundError: If ``build_features_dataset()`` has not been called.
        """
        path = self._cfg.processed_dir / "features_dataset.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Features dataset not found at {path}.  "
                "Run DataPipeline.build_features_dataset() first."
            )
        return _load_parquet(path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_or_fetch_years(
        self,
        name: str,
        years: list[int],
        *,
        force: bool,
    ) -> pd.DataFrame:
        """Return cached data or fetch from ERCOT for each year.

        Per-year Parquet files are stored at ``data/raw/{name}_{year}.parquet``.
        Years with existing files are loaded from disk; missing years are
        fetched and then cached.

        Args:
            name: Dataset type (``"dam_prices"``, ``"rtm_prices"``, or
                  ``"load_actual"``).
            years: Calendar years required.
            force: Skip cache and re-fetch if ``True``.

        Returns:
            Concatenated DataFrame spanning all requested years.
        """
        frames: list[pd.DataFrame] = []
        years_to_fetch: list[int] = []

        for yr in sorted(years):
            path = self._cfg.raw_dir / f"{name}_{yr}.parquet"
            if not force and path.exists():
                logger.info("Cache hit loading {} {} from disk.", name, yr)
                frames.append(_load_parquet(path))
            else:
                years_to_fetch.append(yr)

        if years_to_fetch:
            logger.info("Fetching {} for years={} from ERCOT…", name, years_to_fetch)
            fetched = self._fetch(name, years_to_fetch)

            # Split the fetched data by year and cache each year individually.
            for yr in years_to_fetch:
                start_ts = pd.Timestamp(f"{yr}-01-01", tz="UTC")
                end_ts = pd.Timestamp(f"{yr}-12-31 23:59:59", tz="UTC")
                yr_df = fetched.loc[start_ts:end_ts]
                path = self._cfg.raw_dir / f"{name}_{yr}.parquet"
                _save_parquet(yr_df, path)
                frames.append(yr_df)

        return pd.concat(frames).sort_index()

    def _fetch(self, name: str, years: list[int]) -> pd.DataFrame:
        """Dispatch to the appropriate ``ERCOTClient`` method.

        Args:
            name: Dataset type identifier.
            years: Calendar years to fetch.

        Returns:
            Raw DataFrame from the client.

        Raises:
            ValueError: If ``name`` is not a recognised dataset type.
        """
        match name:
            case "dam_prices":
                df = self._client.get_dam_prices(years)
                self._validate_lmp(df, name, years)
                return df
            case "rtm_prices":
                df = self._client.get_rtm_prices(years)
                self._validate_lmp(df, name, years)
                return df
            case "load_actual":
                return self._client.get_load_actual(years)
            case _:
                raise ValueError(f"Unknown dataset name: {name!r}")

    def _validate_lmp(
        self, df: pd.DataFrame, name: str, years: list[int]
    ) -> None:
        """Run per-year LMP quality checks and log errors without halting the pipeline.

        Validates each year independently so that non-contiguous year lists
        (e.g. [2020, 2021, 2024]) do not incorrectly fail the coverage check
        for the years in between that were not requested.
        """
        for yr in sorted(years):
            yr_start = pd.Timestamp(f"{yr}-01-01", tz="UTC")
            yr_end = pd.Timestamp(f"{yr}-12-31 23:59:59", tz="UTC")
            yr_df = df.loc[yr_start:yr_end]
            try:
                validate_lmp_dataframe(
                    yr_df,
                    start=date(yr, 1, 1),
                    end=date(yr, 12, 31),
                    cfg=self._cfg,
                    label=f"{name.upper()}_{yr}",
                )
            except DataValidationError as exc:
                logger.error("Validation errors for {} {}:\n{}", name, yr, exc)

    def _build_spread_dataset(
        self,
        dam: pd.DataFrame,
        rtm: pd.DataFrame,
        load: pd.DataFrame,
    ) -> pd.DataFrame:
        """Align DAM prices, RTM prices, and load into the spread dataset.

        The merge key is (``interval_start_utc``, ``hub``).  Load is a
        system-wide scalar (not per-hub) joined on timestamp only.

        Args:
            dam: Hourly DAM prices per hub.
            rtm: Hourly RTM prices per hub.
            load: Hourly system-wide actual load.

        Returns:
            Wide-format spread DataFrame.
        """
        # Merge DAM and RTM on (interval_start_utc, hub) to avoid the
        # cross-product that index-only joining produces when there are
        # multiple hubs per timestamp.
        merged = pd.merge(
            dam.rename(columns={"lmp": "dam_lmp"}).reset_index(),
            rtm.rename(columns={"lmp": "rtm_lmp"}).reset_index(),
            on=["interval_start_utc", "hub"],
            how="inner",
        ).set_index("interval_start_utc")

        merged["spread"] = merged["dam_lmp"] - merged["rtm_lmp"]

        # Broadcast system-wide load onto each hub row via index join.
        result = merged.join(load[["load_mw"]], how="left")

        missing_pct = result["load_mw"].isna().mean()
        if missing_pct > 0.05:
            logger.warning(
                "load_mw is missing for {:.1%} of rows after join.", missing_pct
            )

        return result.sort_index()

    def _load_or_fetch_eia_years(
        self,
        name: str,
        years: list[int],
        fetch_fn: object,
        *,
        force: bool,
    ) -> pd.DataFrame:
        """Return cached EIA data or fetch from the API for each year.

        Per-year Parquet files are stored at ``data/raw/{name}_{year}.parquet``.

        Args:
            name: Dataset identifier (e.g. ``"eia_wind"``).
            years: Calendar years required.
            fetch_fn: Callable with signature ``(start: date, end: date) ->
                pd.DataFrame`` one of the ``EIAClient`` methods.
            force: Skip cache and re-fetch if ``True``.

        Returns:
            Concatenated DataFrame spanning all requested years.
        """
        frames: list[pd.DataFrame] = []

        for yr in sorted(years):
            path = self._cfg.raw_dir / f"{name}_{yr}.parquet"
            if not force and path.exists():
                logger.info("Cache hit loading {} {} from disk.", name, yr)
                frames.append(_load_parquet(path))
            else:
                logger.info("Fetching {} for year={} from EIA…", name, yr)
                yr_df = fetch_fn(date(yr, 1, 1), date(yr, 12, 31))
                if not yr_df.empty:
                    _save_parquet(yr_df, path)
                    frames.append(yr_df)
                else:
                    logger.warning("No EIA data returned for {} {}.", name, yr)

        return pd.concat(frames).sort_index() if frames else pd.DataFrame()

    def _try_load_eia(
        self,
        name: str,
        years: list[int],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Load a cached EIA dataset if all requested years are present.

        Returns ``None`` if any year's file is missing (so that
        ``add_eia_features`` simply skips the missing feature group rather
        than erroring).
        """
        frames: list[pd.DataFrame] = []
        for yr in sorted(years):
            path = self._cfg.raw_dir / f"{name}_{yr}.parquet"
            if not path.exists():
                logger.debug("EIA cache miss for {} {} skipping feature group.", name, yr)
                return None
            frames.append(_load_parquet(path))

        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        return combined.loc[start_ts:end_ts] if not combined.empty else None

    def _try_load_weather(
        self,
        years: list[int],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Load cached weather temperature data if all requested years are present.

        Returns ``None`` if any year's file is missing (so that
        ``add_weather_features`` simply skips weather rather than erroring).
        """
        frames: list[pd.DataFrame] = []
        for yr in sorted(years):
            path = self._cfg.raw_dir / f"weather_temperature_{yr}.parquet"
            if not path.exists():
                logger.debug(
                    "Weather cache miss for year {} skipping weather features.", yr
                )
                return None
            frames.append(_load_parquet(path))

        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        return combined.loc[start_ts:end_ts] if not combined.empty else None

    def _try_load_outages(
        self,
        years: list[int],
        start_ts: pd.Timestamp,
        end_ts: pd.Timestamp,
    ) -> pd.DataFrame | None:
        """Load cached outage capacity data if all requested years are present.

        Returns ``None`` if any year's file is missing (so that
        ``add_outage_features`` simply skips the feature group rather than
        erroring).  For the 2020-2024 historical backtest, these files are
        unavailable without an ERCOT API account or manual download, so
        ``None`` is the expected return value.
        """
        frames: list[pd.DataFrame] = []
        for yr in sorted(years):
            path = self._cfg.raw_dir / f"outage_capacity_{yr}.parquet"
            if not path.exists():
                logger.debug(
                    "Outage cache miss for year {} skipping outage features.", yr
                )
                return None
            frames.append(_load_parquet(path))

        if not frames:
            return None

        combined = pd.concat(frames).sort_index()
        return combined.loc[start_ts:end_ts] if not combined.empty else None
