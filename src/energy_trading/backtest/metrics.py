"""Financial performance metrics for the DA/RT spread backtest.

All metrics operate on a per-hour net P&L series (pd.Series of float64).
Zero entries represent hours where no position was held; they are included
in the Sharpe/Sortino denominators but excluded from the win-rate numerator
and denominator.

Metric definitions
------------------
total_pnl          Sum of net_pnl over all OOS hours (dollars).
annualized_pnl     total_pnl scaled to a full trading year:
                   total_pnl * hours_per_year / n_obs.
sharpe             mean(pnl) * sqrt(hours_per_year) / std(pnl, ddof=1).
                   NaN when std == 0 or n_obs < 2.
sortino            mean(pnl) * sqrt(hours_per_year) / semi_std, where
                   semi_std = sqrt(mean(min(pnl, 0) ** 2)).
                   NaN when semi_std == 0 (no downside hours).
max_drawdown       max(cumsum.cummax() - cumsum).  Always >= 0.
win_rate           Fraction of *trading hours* (pnl != 0) where pnl > 0.
                   NaN when n_trades == 0.
                   Note: a trade with exactly net_pnl = 0 is classified as
                   a non-trade by this mask; this is an acceptable
                   approximation for this domain.
cvar_5pct          mean(pnl[pnl <= quantile(pnl, 0.05)]).
                   Expected shortfall in the worst 5 percent of hours
                   (typically negative).
n_trades           Count of non-zero pnl hours (trading hours).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger


def compute_pnl_metrics(
    pnl: pd.Series,
    hours_per_year: int = 8760,
) -> dict[str, float]:
    """Compute a standard set of financial performance metrics on hourly P&L.

    A single input series is used for all metrics.  Zero entries represent
    hours where no position was held; they are included in the Sharpe/Sortino
    denominators but excluded from the win-rate computation.

    Args:
        pnl: Per-hour net P&L series (dollars).  Zeros represent flat hours.
            May contain NaN NaN entries are excluded from all metrics.
        hours_per_year: Annualisation constant.  Default: 8760 (hourly data).

    Returns:
        Dictionary with keys ``total_pnl``, ``annualized_pnl``, ``sharpe``,
        ``sortino``, ``max_drawdown``, ``win_rate``, ``cvar_5pct``,
        ``n_trades``.  All values are Python ``float`` (or ``int`` for
        ``n_trades``).  If the series is empty, all values are ``nan``
        (except ``n_trades`` which is ``0``).
    """
    arr = np.asarray(pnl, dtype=np.float64)

    # Drop NaN entries uniformly
    arr = arr[~np.isnan(arr)]
    n_obs = len(arr)

    nan = float("nan")

    if n_obs == 0:
        logger.warning(
            "compute_pnl_metrics: empty (or all-NaN) series returning NaN for all metrics."
        )
        return dict(
            total_pnl=nan,
            annualized_pnl=nan,
            sharpe=nan,
            sortino=nan,
            max_drawdown=nan,
            win_rate=nan,
            cvar_5pct=nan,
            n_trades=0,
        )

    # --- Basic aggregates ---
    total_pnl = float(arr.sum())
    annualized_pnl = total_pnl * hours_per_year / n_obs
    n_trades = int((arr != 0.0).sum())
    mu = float(arr.mean())

    # --- Sharpe ---
    if n_obs < 2:
        sharpe = nan
    else:
        sigma = float(np.std(arr, ddof=1))
        sharpe = float(mu * np.sqrt(hours_per_year) / sigma) if sigma > 0.0 else nan

    # --- Sortino (semi-deviation denominator) ---
    downside = np.minimum(arr, 0.0)
    semi_var = float(np.mean(downside**2))
    sortino = float(mu * np.sqrt(hours_per_year) / np.sqrt(semi_var)) if semi_var > 0.0 else nan

    # --- Max drawdown ---
    cumsum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cumsum)
    max_drawdown = float(np.max(running_max - cumsum))  # always >= 0

    # --- Win rate (trading hours only) ---
    if n_trades == 0:
        win_rate = nan
    else:
        trading_mask = arr != 0.0
        win_rate = float((arr[trading_mask] > 0.0).mean())

    # --- CVaR 5% (expected shortfall) ---
    q5 = float(np.quantile(arr, 0.05))
    tail = arr[arr <= q5]
    cvar_5pct = float(tail.mean()) if len(tail) > 0 else nan

    logger.debug(
        "Metrics (n_obs={}, n_trades={}): total_pnl={:.2f}  ann_pnl={:.2f}  "
        "Sharpe={:.4f}  Sortino={:.4f}  MaxDD={:.2f}  WR={:.4f}  CVaR={:.2f}",
        n_obs,
        n_trades,
        total_pnl,
        annualized_pnl,
        sharpe,
        sortino,
        max_drawdown,
        win_rate,
        cvar_5pct,
    )

    return dict(
        total_pnl=total_pnl,
        annualized_pnl=annualized_pnl,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        cvar_5pct=cvar_5pct,
        n_trades=n_trades,
    )
