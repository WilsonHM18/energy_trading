"""Risk analysis functions for the ERCOT DA/RT virtual spread backtest.

All functions are pure (no disk I/O).  They operate on the trade DataFrame
produced by ``run_backtest()`` and on the features DataFrame produced by the
data pipeline.

Functions
---------
identify_drawdown_periods   -- find contiguous below-high-water-mark periods
compute_regime_metrics      -- P&L decomposition by market regime
compute_tail_risk           -- VaR, CVaR, skewness, kurtosis, max loss run
tc_sensitivity              -- transaction-cost breakeven sweep
compute_rolling_sharpe      -- rolling annualised Sharpe ratio
parameter_robustness        -- grid search over threshold × lot-size
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from loguru import logger

from energy_trading.backtest.engine import run_backtest
from energy_trading.backtest.metrics import compute_pnl_metrics

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_DD_COLUMNS = [
    "start_date",
    "trough_date",
    "end_date",
    "depth_dollars",
    "duration_hours",
    "recovery_hours",
    "recovered",
]

_REGIME_METRIC_COLUMNS = [
    "regime",
    "n_obs",
    "n_trades",
    "total_pnl",
    "annualized_pnl",
    "sharpe",
    "sortino",
    "max_drawdown",
    "win_rate",
    "cvar_5pct",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def identify_drawdown_periods(cumulative_pnl: pd.Series) -> pd.DataFrame:
    """Identify contiguous drawdown periods from a cumulative P&L series.

    A drawdown period begins when cumulative P&L falls strictly below its
    running maximum (the high-water mark) and ends when it recovers to or
    exceeds that level again.

    Args:
        cumulative_pnl: Cumulative net P&L series (dollars).  Expected to be
            the ``cumulative_pnl`` column from ``compute_hourly_pnl()``
            output.  NaN values are not permitted and will produce incorrect
            results if present.

    Returns:
        ``pd.DataFrame`` with one row per drawdown period, sorted by
        ``depth_dollars`` descending (worst first).  Columns:

        - ``start_date``      timestamp of the last high-water-mark peak
          before the drawdown begins.
        - ``trough_date``     timestamp of the minimum within the period.
        - ``end_date``        timestamp of recovery (first hour at or above
          the prior peak), or the last available timestamp if still
          unrecovered at series end.
        - ``depth_dollars``   magnitude of the trough below the peak
          (always positive).
        - ``duration_hours``  hours from ``start_date`` to ``trough_date``.
        - ``recovery_hours``  hours from ``trough_date`` to ``end_date``
          (``NaN`` if unrecovered).
        - ``recovered``       ``True`` if the series recovered before its end.

        Returns an empty DataFrame (correct columns, zero rows) when there
        are no drawdowns or the input is empty.
    """
    empty = pd.DataFrame(columns=_DD_COLUMNS)

    if len(cumulative_pnl) == 0:
        return empty

    arr = cumulative_pnl.to_numpy(dtype=np.float64)
    idx = cumulative_pnl.index
    n = len(arr)

    running_max = np.maximum.accumulate(arr)
    in_dd = (arr < running_max).astype(np.int8)

    # Run-length encoding: detect transitions into / out of drawdown
    changes = np.diff(in_dd, prepend=np.int8(0), append=np.int8(0))
    starts = np.where(changes == 1)[0]   # first index strictly below peak
    ends = np.where(changes == -1)[0]    # first recovery index (exclusive)

    if len(starts) == 0:
        return empty

    rows: list[dict] = []
    for start, end in zip(starts, ends):
        peak_val = running_max[start]

        # Peak was at the last point before `start` where arr == peak_val
        prior = arr[:start]
        candidates = np.where(prior == peak_val)[0]
        peak_idx = int(candidates[-1]) if len(candidates) > 0 else int(start)

        # Trough is the minimum within [start, end)
        segment = arr[start:end]
        trough_relative = int(np.argmin(segment))
        trough_idx = start + trough_relative
        depth = float(peak_val - arr[trough_idx])

        recovered = bool(end < n)
        end_idx = int(end) if recovered else n - 1
        recovery_hours = float(end_idx - trough_idx) if recovered else float("nan")

        rows.append(
            {
                "start_date": idx[peak_idx],
                "trough_date": idx[trough_idx],
                "end_date": idx[end_idx],
                "depth_dollars": depth,
                "duration_hours": float(trough_idx - peak_idx),
                "recovery_hours": recovery_hours,
                "recovered": recovered,
            }
        )

    result = pd.DataFrame(rows, columns=_DD_COLUMNS)
    result = result.sort_values("depth_dollars", ascending=False).reset_index(drop=True)

    logger.info(
        "identify_drawdown_periods: {} periods found; worst depth={:.2f}, "
        "recovered={}",
        len(result),
        float(result["depth_dollars"].iloc[0]) if len(result) > 0 else float("nan"),
        int(result["recovered"].sum()),
    )
    return result


def compute_regime_metrics(
    trade_df: pd.DataFrame,
    features_df: pd.DataFrame,
    hub: str,
) -> pd.DataFrame:
    """Decompose P&L performance by market regime.

    Joins the per-hour trade results with the features dataset to identify
    regime membership, then calls ``compute_pnl_metrics`` on each subset.

    The ``trade_df`` must be single-hub (as returned by
    ``run_backtest(..., hub=hub)``).  The ``features_df`` is the full
    multi-hub features dataset; it is filtered to the requested hub
    internally before joining.

    Args:
        trade_df: Per-hour trade DataFrame from ``run_backtest``.  Must
            contain a ``net_pnl`` column and a UTC ``DatetimeIndex`` named
            ``interval_start_utc``.
        features_df: Full features dataset (all hubs).  Must contain a
            ``"hub"`` column and feature columns used for regime
            classification (``is_peak``, ``spread_vol_24h``, and optionally
            ``wind_actual_mw``, ``gas_price_mmbtu``).
        hub: Hub identifier, e.g. ``"HB_SOUTH"``.  Used to filter
            ``features_df`` before joining.

    Returns:
        ``pd.DataFrame`` with one row per regime and columns:
        ``regime, n_obs, n_trades, total_pnl, annualized_pnl, sharpe,
        sortino, max_drawdown, win_rate, cvar_5pct``.

    Notes:
        - Optional EIA columns (``wind_actual_mw``, ``gas_price_mmbtu``) are
          skipped with a ``logger.warning`` when absent from ``features_df``.
        - The ``uri_storm`` regime is defined as the interval
          [2021-02-10 UTC, 2021-02-21 UTC).
    """
    # Filter features to single hub → unique DatetimeIndex
    features_hub = features_df[features_df["hub"] == hub].copy()

    # Columns we want to join (only those present)
    wanted_cols = ["is_peak", "spread_vol_24h", "wind_actual_mw", "gas_price_mmbtu"]
    available_cols = [c for c in wanted_cols if c in features_hub.columns]
    missing_cols = set(wanted_cols) - set(available_cols)
    if missing_cols:
        logger.warning(
            "compute_regime_metrics: features_df missing columns {} "
            "corresponding wind/gas regimes will be skipped.",
            sorted(missing_cols),
        )

    joined = trade_df.join(features_hub[available_cols], how="left")

    rows: list[dict] = []

    def _add_regime(name: str, mask: pd.Series) -> None:
        subset = joined.loc[mask, "net_pnl"]
        metrics = compute_pnl_metrics(subset)
        rows.append(
            {
                "regime": name,
                "n_obs": int(mask.sum()),
                **{k: metrics[k] for k in _REGIME_METRIC_COLUMNS[2:]},
            }
        )

    # --- Core regimes (always computed) ---
    all_mask = pd.Series(True, index=joined.index)
    _add_regime("all", all_mask)

    if "is_peak" in joined.columns:
        _add_regime("peak", joined["is_peak"] == 1)
        _add_regime("off_peak", joined["is_peak"] == 0)
    else:
        logger.warning("compute_regime_metrics: 'is_peak' column absent skipping peak regimes.")

    # --- Uri winter storm (Feb 2021) ---
    uri_start = pd.Timestamp("2021-02-10", tz="UTC")
    uri_end = pd.Timestamp("2021-02-21", tz="UTC")
    uri_mask = (joined.index >= uri_start) & (joined.index < uri_end)
    if uri_mask.any():
        _add_regime("uri_storm", uri_mask)
        _add_regime("non_uri", ~uri_mask)
    else:
        logger.warning(
            "compute_regime_metrics: no rows in Uri storm window "
            "(2021-02-10 to 2021-02-21) OOS data may not include 2021."
        )

    # --- Volatility regime ---
    if "spread_vol_24h" in joined.columns:
        vol = joined["spread_vol_24h"].dropna()
        if len(vol) > 0:
            vol_median = float(vol.median())
            _add_regime("high_vol", joined["spread_vol_24h"] >= vol_median)
            _add_regime("low_vol", joined["spread_vol_24h"] < vol_median)

    # --- Wind regime (optional EIA column) ---
    if "wind_actual_mw" in joined.columns:
        wind = joined["wind_actual_mw"].dropna()
        if len(wind) > 0:
            wind_median = float(wind.median())
            _add_regime("high_wind", joined["wind_actual_mw"] >= wind_median)
            _add_regime("low_wind", joined["wind_actual_mw"] < wind_median)

    # --- Gas price regime (optional EIA column) ---
    if "gas_price_mmbtu" in joined.columns:
        gas = joined["gas_price_mmbtu"].dropna()
        if len(gas) > 0:
            gas_median = float(gas.median())
            _add_regime("high_gas", joined["gas_price_mmbtu"] >= gas_median)
            _add_regime("low_gas", joined["gas_price_mmbtu"] < gas_median)

    result = pd.DataFrame(rows, columns=_REGIME_METRIC_COLUMNS)

    logger.info(
        "compute_regime_metrics: hub={}, {} regimes computed.",
        hub,
        len(result),
    )
    return result


def compute_tail_risk(
    net_pnl: pd.Series,
    var_levels: list[float] | None = None,
) -> dict[str, float]:
    """Compute extended tail risk metrics on a net P&L series.

    Extends ``compute_pnl_metrics`` with VaR, CVaR at multiple confidence
    levels, distributional shape statistics, and a maximum consecutive loss
    run metric.  All formulas use numpy only (no scipy dependency).

    Args:
        net_pnl: Per-hour net P&L series (dollars).  Zeros represent flat
            hours.  NaN entries are excluded before computation.
        var_levels: List of quantile levels in (0, 1) for VaR/CVaR
            computation.  Default: ``[0.01, 0.05, 0.10]``.

    Returns:
        Dictionary with keys (all values are Python ``float``):

        - ``var_01pct``, ``var_05pct``, ``var_10pct`` Value-at-Risk at
          1%, 5%, 10% (negative = expected loss at that quantile).
        - ``cvar_01pct``, ``cvar_05pct``, ``cvar_10pct`` Conditional
          VaR (expected shortfall): mean of hours at or below the VaR.
        - ``skewness`` Standardised 3rd moment
          (``mean((x-mu)^3) / std^3``).
        - ``excess_kurtosis`` Standardised 4th moment minus 3
          (``mean((x-mu)^4) / std^4 - 3``).
        - ``max_consecutive_loss_hours`` Longest run of consecutive
          hours with ``net_pnl < 0``.

        Empty or all-NaN input returns all ``nan`` (``0`` for the integer
        ``max_consecutive_loss_hours`` key).
    """
    if var_levels is None:
        var_levels = [0.01, 0.05, 0.10]

    # Level → label suffix mapping (1pct, 5pct, 10pct, ...)
    level_labels = {lvl: f"{int(round(lvl * 100)):02d}pct" for lvl in var_levels}

    arr = np.asarray(net_pnl, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    nan = float("nan")

    empty_result: dict[str, float] = {}
    for lvl in var_levels:
        lbl = level_labels[lvl]
        empty_result[f"var_{lbl}"] = nan
        empty_result[f"cvar_{lbl}"] = nan
    empty_result["skewness"] = nan
    empty_result["excess_kurtosis"] = nan
    empty_result["max_consecutive_loss_hours"] = 0.0

    if len(arr) == 0:
        logger.warning(
            "compute_tail_risk: empty (or all-NaN) series returning NaN for all metrics."
        )
        return empty_result

    result: dict[str, float] = {}

    # --- VaR and CVaR ---
    for lvl in var_levels:
        lbl = level_labels[lvl]
        var_val = float(np.quantile(arr, lvl))
        tail = arr[arr <= var_val]
        cvar_val = float(tail.mean()) if len(tail) > 0 else nan
        result[f"var_{lbl}"] = var_val
        result[f"cvar_{lbl}"] = cvar_val

    # --- Shape statistics (ddof=0: population moments) ---
    mu = float(arr.mean())
    sigma = float(np.std(arr, ddof=0))

    if sigma == 0.0:
        result["skewness"] = nan
        result["excess_kurtosis"] = nan
    else:
        centred = arr - mu
        result["skewness"] = float(np.mean(centred**3) / sigma**3)
        result["excess_kurtosis"] = float(np.mean(centred**4) / sigma**4 - 3.0)

    # --- Maximum consecutive loss run ---
    loss_mask = (arr < 0.0).astype(np.int8)
    if loss_mask.sum() == 0:
        result["max_consecutive_loss_hours"] = 0.0
    else:
        changes = np.diff(loss_mask, prepend=np.int8(0), append=np.int8(0))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        run_lengths = run_ends - run_starts
        result["max_consecutive_loss_hours"] = float(run_lengths.max())

    logger.debug(
        "compute_tail_risk: n={}, skewness={:.4f}, excess_kurtosis={:.4f}, "
        "max_consec_loss={}",
        len(arr),
        result.get("skewness", nan),
        result.get("excess_kurtosis", nan),
        result.get("max_consecutive_loss_hours", 0),
    )
    return result


def tc_sensitivity(
    oos_df: pd.DataFrame,
    model: str,
    hub: str,
    tc_values: list[float],
    threshold: float = 0.0,
    lot_mwh: float = 1.0,
) -> pd.DataFrame:
    """Sweep transaction-cost values and measure P&L degradation.

    Calls ``run_backtest`` for each transaction-cost value and collects
    performance metrics.  Useful for identifying the breakeven transaction
    cost above which the strategy becomes unprofitable.

    Args:
        oos_df: Out-of-sample predictions DataFrame.
        model: Model name, e.g. ``"lgbm"``.
        hub: Hub identifier, e.g. ``"HB_SOUTH"``.
        tc_values: List of round-trip transaction costs ($/MWh) to evaluate.
            Must all be >= 0.
        threshold: Signal threshold (minimum |prediction| to trade).
            Default: 0.0.
        lot_mwh: Position size per trade (MWh).  Default: 1.0.

    Returns:
        ``pd.DataFrame`` sorted by ``tc`` ascending with columns:
        ``tc, total_pnl, annualized_pnl, sharpe, sortino, max_drawdown,
        win_rate, cvar_5pct, n_trades, trade_rate``.
    """
    rows: list[dict] = []
    for tc in tc_values:
        _, metrics = run_backtest(
            oos_df,
            model=model,
            hub=hub,
            threshold=threshold,
            lot_mwh=lot_mwh,
            transaction_cost_per_mwh=tc,
        )
        rows.append(
            {
                "tc": tc,
                "total_pnl": metrics["total_pnl"],
                "annualized_pnl": metrics["annualized_pnl"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "cvar_5pct": metrics["cvar_5pct"],
                "n_trades": metrics["n_trades"],
                "trade_rate": metrics["trade_rate"],
            }
        )

    result = pd.DataFrame(rows).sort_values("tc").reset_index(drop=True)

    logger.info(
        "tc_sensitivity: model={}, hub={}, {} TC values evaluated; "
        "breakeven ~${}",
        model,
        hub,
        len(tc_values),
        _estimate_breakeven_tc(result),
    )
    return result


def _estimate_breakeven_tc(tc_df: pd.DataFrame) -> str:
    """Return a string estimate of the breakeven TC (where total_pnl crosses 0)."""
    if "total_pnl" not in tc_df.columns or len(tc_df) == 0:
        return "unknown"
    profitable = tc_df[tc_df["total_pnl"] > 0]
    if len(profitable) == 0:
        return "<= 0"
    max_profitable_tc = float(profitable["tc"].max())
    return f"~${max_profitable_tc:.2f}/MWh"


def compute_rolling_sharpe(
    net_pnl: pd.Series,
    window_hours: int = 720,
    hours_per_year: int = 8760,
) -> pd.Series:
    """Compute rolling annualised Sharpe ratio over a moving window.

    Args:
        net_pnl: Per-hour net P&L series (dollars).
        window_hours: Rolling window size in hours.  Default: 720 (30 days).
        hours_per_year: Annualisation constant.  Default: 8760.

    Returns:
        ``pd.Series`` of rolling Sharpe ratios with the same index as
        ``net_pnl``.  The first ``window_hours - 1`` entries are ``NaN``
        (insufficient window).  Windows where ``std == 0`` return ``NaN``.
        Series name is ``"rolling_sharpe"``.
    """
    scale = float(np.sqrt(hours_per_year))

    def _sharpe_window(x: np.ndarray) -> float:
        s = float(np.std(x, ddof=1))
        if s == 0.0:
            return float("nan")
        return float(x.mean() * scale / s)

    result = net_pnl.rolling(window_hours).apply(_sharpe_window, raw=True)
    result.name = "rolling_sharpe"

    logger.debug(
        "compute_rolling_sharpe: window={}, n_valid={}, mean_sharpe={:.4f}",
        window_hours,
        int(result.notna().sum()),
        float(result.mean()) if result.notna().any() else float("nan"),
    )
    return result


def parameter_robustness(
    oos_df: pd.DataFrame,
    model: str,
    hub: str,
    thresholds: list[float],
    lot_sizes: list[float] | None = None,
) -> pd.DataFrame:
    """Grid search over signal thresholds and position sizes.

    Evaluates how key performance metrics change as the signal confidence
    threshold and base position size are varied.

    Args:
        oos_df: Out-of-sample predictions DataFrame.
        model: Model name, e.g. ``"lgbm"``.
        hub: Hub identifier, e.g. ``"HB_SOUTH"``.
        thresholds: List of signal thresholds (minimum |prediction|) to
            sweep.  All values must be >= 0.
        lot_sizes: List of base position sizes (MWh).  Default: ``[1.0]``.

    Returns:
        ``pd.DataFrame`` sorted by ``threshold`` then ``lot_mwh`` with
        columns: ``threshold, lot_mwh, total_pnl, annualized_pnl, sharpe,
        sortino, max_drawdown, win_rate, cvar_5pct, n_trades, trade_rate``.
    """
    if lot_sizes is None:
        lot_sizes = [1.0]

    rows: list[dict] = []
    for threshold, lot_mwh in itertools.product(thresholds, lot_sizes):
        _, metrics = run_backtest(
            oos_df,
            model=model,
            hub=hub,
            threshold=threshold,
            lot_mwh=lot_mwh,
        )
        rows.append(
            {
                "threshold": threshold,
                "lot_mwh": lot_mwh,
                "total_pnl": metrics["total_pnl"],
                "annualized_pnl": metrics["annualized_pnl"],
                "sharpe": metrics["sharpe"],
                "sortino": metrics["sortino"],
                "max_drawdown": metrics["max_drawdown"],
                "win_rate": metrics["win_rate"],
                "cvar_5pct": metrics["cvar_5pct"],
                "n_trades": metrics["n_trades"],
                "trade_rate": metrics["trade_rate"],
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values(["threshold", "lot_mwh"])
        .reset_index(drop=True)
    )

    logger.info(
        "parameter_robustness: model={}, hub={}, {} combinations ({} thresholds × {} lot sizes).",
        model,
        hub,
        len(rows),
        len(thresholds),
        len(lot_sizes),
    )
    return result
