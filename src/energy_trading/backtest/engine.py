"""Backtest engine for ERCOT DA/RT virtual spread trading.

All functions are pure (no class state, no disk I/O).  They consume the
out-of-sample predictions DataFrame produced by ``run_walk_forward()`` and
written to ``data/processed/oos_predictions.parquet``.

OOS DataFrame schema
--------------------
Index : interval_start_utc  (UTC DatetimeIndex, hourly, **non-unique**
        4 hubs share identical timestamps)
hub                          str  one of HB_NORTH/SOUTH/WEST/HOUSTON
spread                       float64  realised DAM LMP − RTM LMP ($/MWh)
spread_pred_linear           float64
spread_pred_ridge            float64
spread_pred_lasso            float64
spread_pred_lgbm             float64

Trading convention
------------------
signal = +1  →  long spread  (buy DA, sell RT): profit when spread > 0
signal = -1  →  short spread (sell DA, buy RT): profit when spread < 0
signal =  0  →  flat (no trade)
gross_pnl    =  signal × position_mwh × spread_actual  (dollars)

Non-unique index alignment
--------------------------
When ``hub=None`` the OOS DataFrame retains its non-unique DatetimeIndex
(4 rows per timestamp).  ``generate_signals`` preserves this index in its
output.  ``compute_hourly_pnl`` aligns rows via ``oos_df.loc[signals.index]``,
which is safe because ``signals.index`` is a positionally identical slice of
the same DataFrame.  **Never use pd.merge or DataFrame.join on this index**
doing so creates a cross-product (known bug from Phase 2 data work).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from loguru import logger

from energy_trading.backtest.metrics import compute_pnl_metrics


def generate_signals(
    oos_df: pd.DataFrame,
    model: str,
    threshold: float = 0.0,
    hub: str | None = None,
) -> pd.Series:
    """Generate integer trading signals from model predictions.

    Args:
        oos_df: Out-of-sample predictions DataFrame.  Must contain a column
            ``spread_pred_{model}`` and a ``"hub"`` column.
        model: Model name, e.g. ``"linear"``, ``"lgbm"``.  Used to form the
            prediction column name ``spread_pred_{model}``.
        threshold: Minimum absolute prediction required to trade.  Must be
            >= 0.  Predictions in ``[-threshold, +threshold]`` produce a
            flat (0) signal.  Default: 0.0 (trade on every nonzero prediction).
        hub: If given, only rows where ``oos_df["hub"] == hub`` are included.
            If ``None``, all hubs are included (portfolio view).

    Returns:
        ``pd.Series`` of ``int8`` with values in ``{-1, 0, +1}`` and the
        same index as the (filtered) ``oos_df``.  Name is ``"signal"``.

    Raises:
        ValueError: If ``threshold < 0``.
        ValueError: If ``spread_pred_{model}`` is not a column of ``oos_df``.
        ValueError: If ``hub`` is given but not present in ``oos_df["hub"]``.
    """
    if threshold < 0.0:
        raise ValueError(f"threshold must be >= 0, got {threshold}.")

    pred_col = f"spread_pred_{model}"
    if pred_col not in oos_df.columns:
        raise ValueError(
            f"Model column {pred_col!r} not found in oos_df. "
            f"Available columns: {sorted(c for c in oos_df.columns if c.startswith('spread_pred_'))}"
        )

    if hub is not None:
        if hub not in oos_df["hub"].values:
            raise ValueError(
                f"Hub {hub!r} not found in oos_df. "
                f"Available hubs: {sorted(oos_df['hub'].unique().tolist())}"
            )
        df = oos_df[oos_df["hub"] == hub]
    else:
        df = oos_df

    pred = df[pred_col].to_numpy(dtype=np.float64)

    signals = np.zeros(len(pred), dtype=np.int8)
    signals[pred > threshold] = np.int8(1)
    signals[pred < -threshold] = np.int8(-1)
    # NaN comparisons with > and < return False in numpy → NaN preds → signal=0 (flat). Correct.

    result = pd.Series(signals, index=df.index, name="signal", dtype=np.int8)

    logger.debug(
        "generate_signals: model={}, hub={}, threshold={}, n_rows={}, n_trades={}",
        model,
        hub,
        threshold,
        len(result),
        int((signals != 0).sum()),
    )
    return result


def compute_hourly_pnl(
    oos_df: pd.DataFrame,
    signals: pd.Series,
    lot_mwh: float = 1.0,
    transaction_cost_per_mwh: float = 0.0,
    max_mwh: float = 100.0,
    scale_by_confidence: bool = False,
    model: str | None = None,
) -> pd.DataFrame:
    """Compute per-hour P&L given trading signals and position parameters.

    Args:
        oos_df: Same OOS DataFrame passed to ``generate_signals``.
        signals: Signal series returned by ``generate_signals``.  Must have
            an index that is a positionally identical slice of ``oos_df.index``
            (guaranteed when both use the same source DataFrame and hub filter).
        lot_mwh: Base position size in MWh per trade.  Default: 1.0 MWh.
        transaction_cost_per_mwh: Round-trip transaction cost per MWh traded.
            Applied as ``|signal| × position_mwh × tc``.  Default: 0.0.
        max_mwh: Position cap in MWh (ERCOT-style limit).  Default: 100.0.
        scale_by_confidence: If ``True``, scale ``lot_mwh`` by
            ``|pred| / mean(|pred|)`` before applying the cap.  Requires
            ``model`` to be specified.  Default: ``False``.
        model: Model name used to locate the prediction column when
            ``scale_by_confidence=True`` or to populate ``spread_pred`` column.
            If ``None`` and ``scale_by_confidence=True``, raises ``ValueError``.

    Returns:
        ``pd.DataFrame`` with the same index as ``signals`` and columns:
        ``signal``, ``position_mwh``, ``spread_pred``, ``spread_actual``,
        ``gross_pnl``, ``transaction_cost_paid``, ``net_pnl``,
        ``cumulative_pnl``.

    Raises:
        ValueError: If ``scale_by_confidence=True`` and ``model`` is ``None``.
    """
    if scale_by_confidence and model is None:
        raise ValueError(
            "model must be specified when scale_by_confidence=True."
        )

    # oos_df must be the same source DataFrame that was used to generate signals
    # (same rows, same order).  Never use .loc or .join for row alignment here:
    # when the DatetimeIndex is non-unique (4 hubs × same timestamps), .loc
    # produces a cross-product and returns far more rows than expected.
    # run_backtest pre-filters oos_df by hub before calling this function,
    # ensuring that len(oos_df) == len(signals) is always satisfied.
    if len(oos_df) != len(signals):
        raise ValueError(
            f"oos_df (n={len(oos_df)}) and signals (n={len(signals)}) must have the "
            "same length.  Pass the hub-filtered source DataFrame that was used to "
            "generate signals, not the full multi-hub OOS DataFrame."
        )
    df = oos_df

    sig = signals.to_numpy(dtype=np.int8)
    spread_actual = df["spread"].to_numpy(dtype=np.float64)
    n = len(sig)

    # --- Position sizing ---
    position_mwh = np.full(n, lot_mwh, dtype=np.float64)

    if scale_by_confidence:
        pred_col = f"spread_pred_{model}"
        abs_pred = np.abs(df[pred_col].to_numpy(dtype=np.float64))
        mean_abs = float(np.nanmean(abs_pred))
        if mean_abs == 0.0:
            logger.warning(
                "compute_hourly_pnl: mean(|pred|) == 0 for model={} "
                "skipping confidence scaling.",
                model,
            )
        else:
            position_mwh = position_mwh * (abs_pred / mean_abs)

    # Apply position cap
    position_mwh = np.minimum(position_mwh, max_mwh)
    # Zero out position when flat (signal == 0)
    position_mwh = np.where(sig != 0, position_mwh, 0.0)

    # --- P&L computation ---
    sig_f = sig.astype(np.float64)
    gross_pnl = sig_f * position_mwh * spread_actual
    transaction_cost_paid = np.abs(sig_f) * position_mwh * transaction_cost_per_mwh
    net_pnl = gross_pnl - transaction_cost_paid
    cumulative_pnl = np.cumsum(net_pnl)

    # --- Spread prediction column ---
    if model is not None:
        pred_col = f"spread_pred_{model}"
        spread_pred = df[pred_col].to_numpy(dtype=np.float64)
    else:
        spread_pred = np.full(n, np.nan)

    result = pd.DataFrame(
        {
            "signal": sig,
            "position_mwh": position_mwh,
            "spread_pred": spread_pred,
            "spread_actual": spread_actual,
            "gross_pnl": gross_pnl,
            "transaction_cost_paid": transaction_cost_paid,
            "net_pnl": net_pnl,
            "cumulative_pnl": cumulative_pnl,
        },
        index=signals.index,
    )

    logger.debug(
        "compute_hourly_pnl: n_rows={}, n_trades={}, total_net_pnl={:.2f}",
        n,
        int((sig != 0).sum()),
        float(net_pnl.sum()),
    )
    return result


def run_backtest(
    oos_df: pd.DataFrame,
    model: str,
    hub: str | None = None,
    threshold: float = 0.0,
    lot_mwh: float = 1.0,
    transaction_cost_per_mwh: float = 0.0,
    max_mwh: float = 100.0,
    scale_by_confidence: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run a complete backtest for one (model, hub, threshold) combination.

    Chains ``generate_signals`` → ``compute_hourly_pnl`` →
    ``compute_pnl_metrics`` into a single end-to-end call.

    Args:
        oos_df: Out-of-sample predictions DataFrame.
        model: Model name, e.g. ``"linear"``, ``"lgbm"``.
        hub: If given, restrict to a single hub.  If ``None``, trade all hubs
            simultaneously (portfolio view).
        threshold: Minimum |prediction| to trigger a trade.  Default: 0.0.
        lot_mwh: Base position size per trade (MWh).  Default: 1.0.
        transaction_cost_per_mwh: Round-trip transaction cost ($/MWh).
            Default: 0.0.
        max_mwh: Position cap per hour (MWh).  Default: 100.0.
        scale_by_confidence: Scale position by |pred| / mean(|pred|).
            Default: ``False``.

    Returns:
        Tuple ``(trade_df, metrics_dict)`` where:

        **trade_df** per-hour DataFrame with columns:
          ``signal, position_mwh, spread_pred, spread_actual, gross_pnl,
          transaction_cost_paid, net_pnl, cumulative_pnl``.

        **metrics_dict** performance summary with keys:
          ``model, hub, threshold, lot_mwh, n_obs, trade_rate,
          n_trades, total_pnl, annualized_pnl, sharpe, sortino,
          max_drawdown, win_rate, cvar_5pct``.
    """
    # Pre-filter to hub before generating signals and computing P&L.
    # This avoids the non-unique DatetimeIndex cross-product bug:
    # .loc on a non-unique index expands each timestamp to all matching hubs.
    if hub is not None:
        source_df = oos_df[oos_df["hub"] == hub].copy()
    else:
        source_df = oos_df  # keep all hubs (still non-unique, but signals match)

    # hub=None: generate_signals sees the already-filtered source_df, so no hub
    # filtering is needed inside generate_signals for this call.
    signals = generate_signals(source_df, model=model, threshold=threshold, hub=None)
    trade_df = compute_hourly_pnl(
        source_df,
        signals,
        lot_mwh=lot_mwh,
        transaction_cost_per_mwh=transaction_cost_per_mwh,
        max_mwh=max_mwh,
        scale_by_confidence=scale_by_confidence,
        model=model,
    )
    pnl_metrics = compute_pnl_metrics(trade_df["net_pnl"])

    n_obs = len(trade_df)
    n_trades = pnl_metrics["n_trades"]
    trade_rate = float(n_trades / n_obs) if n_obs > 0 else float("nan")

    metrics_dict: dict = {
        "model": model,
        "hub": hub if hub is not None else "all",
        "threshold": threshold,
        "lot_mwh": lot_mwh,
        "n_obs": n_obs,
        "trade_rate": trade_rate,
        **pnl_metrics,
    }

    logger.info(
        "run_backtest: model={}, hub={}, threshold={}, n_obs={}, n_trades={}, "
        "total_pnl={:.2f}, sharpe={:.4f}",
        model,
        hub,
        threshold,
        n_obs,
        n_trades,
        pnl_metrics["total_pnl"],
        pnl_metrics["sharpe"],
    )
    return trade_df, metrics_dict


def run_all_backtests(
    oos_df: pd.DataFrame,
    models: list[str],
    thresholds: list[float],
    lot_mwh: float = 1.0,
    transaction_cost_per_mwh: float = 0.0,
    max_mwh: float = 100.0,
    scale_by_confidence: bool = False,
    per_hub: bool = True,
) -> pd.DataFrame:
    """Run a grid of backtests over all (model × hub × threshold) combinations.

    Args:
        oos_df: Out-of-sample predictions DataFrame.
        models: List of model names to evaluate.
        thresholds: List of prediction-magnitude thresholds to sweep.
        lot_mwh: Base position size (MWh).  Default: 1.0.
        transaction_cost_per_mwh: Round-trip transaction cost ($/MWh).
            Default: 0.0.
        max_mwh: Position cap per hour (MWh).  Default: 100.0.
        scale_by_confidence: Scale position by |pred| / mean(|pred|).
            Default: ``False``.
        per_hub: If ``True``, run each hub independently and return one row
            per ``(model, hub, threshold)`` combination.  If ``False``,
            aggregate all hubs together and return one row per
            ``(model, threshold)``.  Default: ``True``.

    Returns:
        ``pd.DataFrame`` with one row per evaluated combination.  Columns
        match the ``metrics_dict`` keys from ``run_backtest``.
        Total rows: ``len(models) × n_hubs × len(thresholds)`` when
        ``per_hub=True``, or ``len(models) × len(thresholds)`` otherwise.
    """
    if per_hub:
        hubs: list[str | None] = sorted(oos_df["hub"].unique().tolist())
    else:
        hubs = [None]

    rows: list[dict] = []
    for model, hub, threshold in itertools.product(models, hubs, thresholds):
        _, metrics = run_backtest(
            oos_df,
            model=model,
            hub=hub,
            threshold=threshold,
            lot_mwh=lot_mwh,
            transaction_cost_per_mwh=transaction_cost_per_mwh,
            max_mwh=max_mwh,
            scale_by_confidence=scale_by_confidence,
        )
        rows.append(metrics)

    logger.info(
        "run_all_backtests complete: {} combinations ({} models × {} hubs × {} thresholds).",
        len(rows),
        len(models),
        len(hubs),
        len(thresholds),
    )
    return pd.DataFrame(rows)
