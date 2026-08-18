"""Feature engineering for the DA/RT spread forecasting model.

All features are constructed to respect **strict time causality**: every
value used to predict hour *t* must be observable before DAM close for the
operating day containing hour *t* (approximately 10:00 AM CPT the prior day,
i.e. ~36 hours before the start of a peak hour).

Feature groups
--------------
1. **Calendar** hour-of-day, day-of-week, month, season, weekend flag,
   ERCOT peak/off-peak flag.  Pure deterministic signals available with zero
   lag.

2. **Lagged spread** spread at the same hub/hour from prior days and weeks.
   Captures mean-reversion and auto-correlation in the spread series.

3. **Lagged price levels** prior-day DAM and RTM LMP.  Captures price-level
   regime context (e.g. high-gas-price periods tend to have larger and more
   volatile spreads).

4. **Load level & ramp** system-wide actual load and hour-over-hour change.
   High load + steep ramp → RTM scarcity → spread compression or sign flip.

5. **Rolling volatility** rolling standard deviation of the spread over the
   preceding 24 h and 168 h (one week).  High volatility regimes reduce
   signal-to-noise and should reduce position sizing.

6. **Hub spread-to-hub spread** cross-hub spread differentials.  Captures
   congestion regime shifts between West (wind-driven) and the other hubs.
   ``west_north_spread_diff``  (HB_WEST spread) − (HB_NORTH spread), lagged 1 day.

Note on look-ahead
------------------
Lag windows are computed with ``shift(n)`` where *n ≥ 1* on an hourly index,
so no future information leaks into any feature.  Rolling windows use
``min_periods=1`` to avoid introducing NaN-gaps at the start of each hub's
history, and ``closed='left'`` (where supported) to exclude the current
observation from the window.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# ERCOT peak-hour definition (HB protocol: HE 7–22 on non-weekend days)
# ---------------------------------------------------------------------------
_PEAK_HOURS = set(range(7, 23))          # Hour-ending 7 through 22 (06:00–21:59 UTC-6)
_CPT_UTC_OFFSET = 6                     # CPT = UTC-6 (winter); approximation for hour_cpt


def build_features(
    spread_df: pd.DataFrame,
    lag_days: list[int] | None = None,
    lag_weeks: list[int] | None = None,
    volatility_windows_h: list[int] | None = None,
) -> pd.DataFrame:
    """Construct the full feature matrix from the spread dataset.

    This is the primary entry point for feature engineering.  It applies all
    feature groups in sequence and returns a single wide DataFrame ready for
    model training.

    The input ``spread_df`` is the output of ``DataPipeline.load_spread_dataset()``.
    Features that require data not present in the spread dataset (wind, solar,
    gas prices) are added separately via ``add_eia_features()``.

    Args:
        spread_df: Spread dataset with columns ``hub``, ``dam_lmp``,
            ``rtm_lmp``, ``spread``, ``load_mw``.  Index is
            ``interval_start_utc`` (UTC, hourly).
        lag_days: Number of prior days to include as spread lags.  Defaults
            to ``[1, 2, 7]``.
        lag_weeks: Number of prior weeks to include as spread lags.  Defaults
            to ``[1, 2, 4]``.
        volatility_windows_h: Rolling window sizes (hours) for spread
            volatility.  Defaults to ``[24, 168]``.

    Returns:
        Feature DataFrame with the same index as ``spread_df`` and all
        engineered columns appended.  The target column ``spread`` is
        retained as the last column.
    """
    lag_days = lag_days or [1, 2, 7]
    lag_weeks = lag_weeks or [1, 2, 4]
    volatility_windows_h = volatility_windows_h or [24, 168]

    logger.info(
        "Building features for {} rows across {} hubs.",
        len(spread_df),
        spread_df["hub"].nunique(),
    )

    df = spread_df.copy()

    df = _add_calendar_features(df)
    df = _add_lagged_spread(df, lag_days=lag_days, lag_weeks=lag_weeks)
    df = _add_lagged_price_levels(df, lag_days=[1, 7])
    df = _add_load_features(df)
    df = _add_rolling_volatility(df, windows_h=volatility_windows_h)
    df = _add_cross_hub_spread(df)

    n_features = df.shape[1] - spread_df.shape[1]
    logger.info("Feature engineering complete: {} new columns added.", n_features)

    return df


def add_eia_features(
    features_df: pd.DataFrame,
    wind_actual: pd.DataFrame | None = None,
    solar_actual: pd.DataFrame | None = None,
    gas_price: pd.DataFrame | None = None,
    load_forecast: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge EIA-sourced features into the feature matrix.

    Each input is optional; pass only the datasets that have been fetched.
    All inputs must share the same UTC hourly DatetimeIndex as ``features_df``.

    Args:
        features_df: Output of ``build_features()``.
        wind_actual: Hourly wind generation (MW) with column ``wind_actual_mw``.
        solar_actual: Hourly solar generation (MW) with column
            ``solar_actual_mw``.
        gas_price: Daily Henry Hub or Houston Ship Channel gas price
            ($/MMBtu) with column ``gas_price_mmbtu``.  Forward-filled to
            hourly.
        load_forecast: DAM load forecast (MW) with column
            ``load_forecast_mw``.

    Returns:
        Feature DataFrame with EIA columns appended.
    """
    df = features_df.copy()

    if wind_actual is not None:
        df = df.join(wind_actual[["wind_actual_mw"]], how="left")
        # Lag by 24 h so only yesterday's wind is used as a predictor.
        df["wind_actual_mw_lag1d"] = (
            df.groupby("hub")["wind_actual_mw"].shift(24)
        )
        logger.info("Added wind_actual_mw features.")

    if solar_actual is not None:
        df = df.join(solar_actual[["solar_actual_mw"]], how="left")
        df["solar_actual_mw_lag1d"] = (
            df.groupby("hub")["solar_actual_mw"].shift(24)
        )
        logger.info("Added solar_actual_mw features.")

    if gas_price is not None:
        # Gas prices are daily; forward-fill to hourly then lag by 1 day.
        # Use the deduplicate-shift-map pattern (same as _add_load_features) so
        # that the 24-step shift refers to 24 *hours*, not 24 *rows*.  The
        # DataFrame has 4 hub rows per timestamp, so a bare shift(24) would only
        # move back 6 hours.
        gas_hourly = gas_price[["gas_price_mmbtu"]].resample("h").ffill()
        df = df.join(gas_hourly, how="left")
        unique_gas = df[~df.index.duplicated(keep="first")]["gas_price_mmbtu"].sort_index()
        df["gas_price_lag1d"] = df.index.map(unique_gas.shift(24))
        # Gas-electric spread proxy: DAM LMP minus gas heat-rate equivalent.
        # Assumes ~7 MMBtu/MWh heat rate for a marginal gas unit.
        if "gas_price_lag1d" in df.columns:
            df["gas_elec_spread"] = df["dam_lmp"] - 7.0 * df["gas_price_lag1d"]
        logger.info("Added gas_price features.")

    if load_forecast is not None:
        df = df.join(load_forecast[["load_forecast_mw"]], how="left")
        # Use load_mw_lag1h (actual load from the prior hour) instead of the
        # current hour's realised load, which is not available at DAM close.
        load_col = "load_mw_lag1h" if "load_mw_lag1h" in df.columns else "load_mw"
        df["load_forecast_error"] = df[load_col] - df["load_forecast_mw"]
        logger.info("Added load_forecast_error features.")


    return df


def add_weather_features(
    features_df: pd.DataFrame,
    temperature: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge weather-sourced features into the feature matrix.

    Temperature is a system-wide scalar (one value per hour, shared across
    all hubs) sourced from the Open-Meteo ERA5 archive via ``WeatherClient``.
    It is joined by UTC timestamp and broadcast to all hub rows automatically.

    Three derived columns are added:

    * ``temperature_c`` raw hourly 2 m temperature (°C).
    * ``temperature_c_lag24h`` prior-day same-hour temperature; this is the
      causally safe predictor (observable at DAM close ~36 h ahead).
    * ``cooling_degree_hours`` ``max(0, temperature_c_lag24h - 18.3)``
      (65 °F threshold).  Non-linear proxy for cooling load pressure.

    Args:
        features_df: Output of ``build_features()`` (and optionally
            ``add_eia_features()``).
        temperature: Hourly temperature DataFrame with UTC DatetimeIndex
            named ``interval_start_utc`` and column ``temperature_c``.
            Pass ``None`` to skip weather features gracefully.

    Returns:
        Feature DataFrame with weather columns appended.
    """
    df = features_df.copy()

    if temperature is not None:
        df = df.join(temperature[["temperature_c"]], how="left")
        # Causal lag: temperature is system-wide so no groupby is needed, but
        # the DataFrame has 4 hub rows per timestamp.  Use the same
        # deduplicate-shift-map pattern as _add_load_features so that shift(24)
        # refers to 24 *hours*, not 24 *rows*.
        unique_temp = df[~df.index.duplicated(keep="first")]["temperature_c"].sort_index()
        df["temperature_c_lag24h"] = df.index.map(unique_temp.shift(24))
        df["cooling_degree_hours"] = (df["temperature_c_lag24h"] - 18.3).clip(lower=0.0)
        logger.info(
            "add_weather_features: added temperature_c, temperature_c_lag24h, "
            "cooling_degree_hours."
        )

    return df


def add_outage_features(
    features_df: pd.DataFrame,
    outage_capacity: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge ERCOT thermal outage capacity features into the feature matrix.

    Outage capacity (NP3-233-CD) is system-wide (one value per hour, shared
    across all hubs).  It is joined by UTC timestamp and broadcast to all hub
    rows automatically.

    Three derived columns are added:

    * ``total_outage_mw``       total thermal MW offline per the Outage
      Scheduler (excludes IRR / wind / solar).
    * ``total_outage_mw_lag1d`` prior-day same-hour outage level; the
      causally safe predictor (shift(24), observable at DAM close).
    * ``outage_change_1d``      day-over-day change in thermal outages
      (``lag1d − lag2d``), capturing trend in generator availability.

    .. note::
        ERCOT's MIS API retains only the last 31 days of outage reports, so
        this feature group is unavailable for the 2020-2024 historical
        backtest without an ERCOT API account (Dec 2023+) or manual bulk
        download.  Pass ``None`` to skip gracefully.

    Args:
        features_df: Output of ``build_features()`` (and optionally
            ``add_eia_features()`` / ``add_weather_features()``).
        outage_capacity: Hourly outage DataFrame with UTC DatetimeIndex
            named ``interval_start_utc`` and column ``total_outage_mw``.
            Pass ``None`` to skip outage features gracefully.

    Returns:
        Feature DataFrame with outage columns appended.
    """
    df = features_df.copy()

    if outage_capacity is not None:
        df = df.join(outage_capacity[["total_outage_mw"]], how="left")
        # Outage is system-wide; use deduplicate-shift-map so shift(24) means
        # 24 hours, not 24 rows (there are 4 hub rows per timestamp).
        unique_outage = df[~df.index.duplicated(keep="first")]["total_outage_mw"].sort_index()
        lag1d = df.index.map(unique_outage.shift(24))
        lag2d = df.index.map(unique_outage.shift(48))
        df["total_outage_mw_lag1d"] = lag1d
        df["outage_change_1d"] = lag1d - lag2d
        logger.info(
            "add_outage_features: added total_outage_mw, "
            "total_outage_mw_lag1d, outage_change_1d."
        )

    return df


# ---------------------------------------------------------------------------
# Private feature-group builders
# ---------------------------------------------------------------------------


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic calendar columns.

    All columns are derived from the UTC index; the CPT offset is applied
    where ERCOT-specific definitions are needed (peak/off-peak).

    New columns
    -----------
    hour_utc, hour_cpt, dow, month, quarter, is_weekend, is_peak,
    sin_hour, cos_hour, sin_month, cos_month
    """
    idx = df.index

    df["hour_utc"] = idx.hour.astype(np.int8)
    # Approximate CPT (UTC-6); does not adjust for CDT (UTC-5) acceptable
    # approximation given the primary signal is hour-of-day shape.
    df["hour_cpt"] = ((idx.hour - _CPT_UTC_OFFSET) % 24).astype(np.int8)
    df["dow"] = idx.dayofweek.astype(np.int8)          # 0=Mon, 6=Sun
    df["month"] = idx.month.astype(np.int8)
    df["quarter"] = idx.quarter.astype(np.int8)
    df["is_weekend"] = (idx.dayofweek >= 5).astype(np.int8)
    df["is_peak"] = (
        (df["hour_cpt"].isin(_PEAK_HOURS)) & (~df["is_weekend"].astype(bool))
    ).astype(np.int8)

    # Cyclical encodings capture the continuous periodicity of hour and month
    # more naturally than integer values for linear models.
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_cpt"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_cpt"] / 24)
    df["sin_month"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["cos_month"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df


def _add_lagged_spread(
    df: pd.DataFrame,
    lag_days: list[int],
    lag_weeks: list[int],
) -> pd.DataFrame:
    """Add lagged spread values at prior days and weeks.

    Lags are computed per-hub using ``groupby`` + ``shift`` on the hourly
    index, so cross-hub contamination is impossible.

    New columns
    -----------
    spread_lag_{n}d  for n in lag_days
    spread_lag_{n}w  for n in lag_weeks
    """
    for n in lag_days:
        df[f"spread_lag_{n}d"] = df.groupby("hub")["spread"].shift(n * 24)
    for n in lag_weeks:
        df[f"spread_lag_{n}w"] = df.groupby("hub")["spread"].shift(n * 168)

    return df


def _add_lagged_price_levels(
    df: pd.DataFrame,
    lag_days: list[int],
) -> pd.DataFrame:
    """Add lagged DAM and RTM price levels.

    New columns
    -----------
    dam_lmp_lag_{n}d, rtm_lmp_lag_{n}d  for n in lag_days
    """
    for n in lag_days:
        df[f"dam_lmp_lag_{n}d"] = df.groupby("hub")["dam_lmp"].shift(n * 24)
        df[f"rtm_lmp_lag_{n}d"] = df.groupby("hub")["rtm_lmp"].shift(n * 24)

    return df


def _add_load_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add load level and ramp features.

    Load is system-wide (not per-hub) so no groupby is needed but we still
    use shift(1) to ensure the current hour's load is not used as a predictor
    of the current hour's spread (load is realised in real-time, not known
    at DAM close).

    New columns
    -----------
    load_mw_lag1h   load from one hour ago (safe proxy for current level)
    load_mw_lag1d   load from same hour yesterday
    load_ramp_1h    hour-over-hour change in load (lag1h - lag2h)
    load_ramp_24h   day-over-day change at the same hour
    """
    # Unique timestamp → load mapping (load is the same for all hubs).
    load_series = df[~df.index.duplicated(keep="first")]["load_mw"].sort_index()

    load_lag1h = load_series.shift(1)
    load_lag2h = load_series.shift(2)
    load_lag24h = load_series.shift(24)
    load_lag48h = load_series.shift(48)

    df["load_mw_lag1h"] = df.index.map(load_lag1h)
    df["load_mw_lag1d"] = df.index.map(load_lag24h)
    df["load_ramp_1h"] = df.index.map(load_lag1h - load_lag2h)
    df["load_ramp_24h"] = df.index.map(load_lag24h - load_lag48h)

    return df


def _add_rolling_volatility(
    df: pd.DataFrame,
    windows_h: list[int],
) -> pd.DataFrame:
    """Add rolling spread volatility (std dev) over prior windows.

    Uses an exclusive right-closed window (``closed='left'`` equivalent via
    ``shift(1)`` before rolling) so the current observation is never included.

    New columns
    -----------
    spread_vol_{n}h  for n in windows_h
    """
    for w in windows_h:
        col = f"spread_vol_{w}h"
        df[col] = (
            df.groupby("hub")["spread"]
            .shift(1)
            .groupby(df["hub"])
            .transform(lambda s: s.rolling(window=w, min_periods=max(1, w // 4)).std())
        )

    return df


def _add_cross_hub_spread(df: pd.DataFrame) -> pd.DataFrame:
    """Add the West-to-North hub spread differential.

    HB_WEST is the most wind-exposed hub.  Its spread relative to HB_NORTH
    (which anchors DFW load) is a proxy for real-time congestion on the
    West-to-North transmission corridor.

    New columns
    -----------
    west_north_spread_diff  (HB_WEST spread) - (HB_NORTH spread), lagged 1 day
    """
    if not {"HB_WEST", "HB_NORTH"}.issubset(df["hub"].unique()):
        return df

    pivot = df[["hub", "spread"]].pivot_table(
        index=df.index, columns="hub", values="spread", aggfunc="first"
    )
    west_north_diff = (pivot["HB_WEST"] - pivot["HB_NORTH"]).shift(24)
    west_north_diff.name = "west_north_spread_diff"

    df = df.join(west_north_diff, how="left")
    return df


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns (excludes raw price/spread columns).

    Args:
        df: Feature DataFrame produced by ``build_features()``.

    Returns:
        Ordered list of column names suitable for use as model inputs.
    """
    raw = {"hub", "dam_lmp", "rtm_lmp", "spread", "load_mw"}
    return [c for c in df.columns if c not in raw]


def drop_warmup_rows(df: pd.DataFrame, min_lag_hours: int = 168) -> pd.DataFrame:
    """Drop the initial rows where lagged features are unavailable.

    The first ``min_lag_hours`` rows per hub will have NaN in the longest-lag
    features (default: 1 week = 168 h).  These rows should be excluded from
    training to avoid biasing the model with NaN-imputed values.

    Args:
        df: Feature DataFrame.
        min_lag_hours: Number of hours to drop from the start of each hub's
            history.  Should equal the longest lag used in feature engineering.

    Returns:
        DataFrame with warmup rows removed.
    """
    cutoff = df.index.min() + pd.Timedelta(hours=min_lag_hours)
    trimmed = df.loc[df.index >= cutoff]
    dropped = len(df) - len(trimmed)
    logger.info(
        "Dropped {} warmup rows (< {} hours from start).", dropped, min_lag_hours
    )
    return trimmed
