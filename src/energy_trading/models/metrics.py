"""Forecast evaluation metrics for the DA/RT spread model.

All metrics are computed on the same NaN-masked rows so that partial
NaN inputs do not silently produce metrics computed on different subsets.

Metric definitions
------------------
rmse               Root-mean-squared error ($/MWh).
mae                Mean absolute error ($/MWh).
r2                 Coefficient of determination (sklearn convention: can be
                   negative when predictions are worse than the mean baseline).
direction_accuracy Fraction of hours where sign(y_pred) == sign(y_true).
                   Zero spreads are treated as a separate sign class, so a
                   zero prediction on a positive-spread hour is a miss.
ic                 Pearson correlation (Information Coefficient).
                   Standard in commodity quant research; measures the linear
                   association between forecast rank and realised spread.
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from scipy import stats


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute a standard set of regression and directional trading metrics.

    A single NaN mask is applied to both arrays before any computation, so
    every returned metric is based on the same set of observations.

    Args:
        y_true: Array of realised spread values ($/MWh).
        y_pred: Array of model predictions, same length as ``y_true``.

    Returns:
        Dictionary with keys ``rmse``, ``mae``, ``r2``,
        ``direction_accuracy``, ``ic``.  All values are Python ``float``.
        If no valid (non-NaN) rows exist, all values are ``nan``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    n_valid = int(valid.sum())

    if n_valid == 0:
        logger.warning(
            "compute_metrics: no valid (non-NaN) rows returning NaN for all metrics."
        )
        nan = float("nan")
        return dict(rmse=nan, mae=nan, r2=nan, direction_accuracy=nan, ic=nan)

    t = y_true[valid]
    p = y_pred[valid]

    # --- Regression metrics ---
    residuals = t - p
    rmse = float(np.sqrt(np.mean(residuals**2)))
    mae = float(np.mean(np.abs(residuals)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    # --- Trading metric: directional accuracy ---
    direction_accuracy = float(np.mean(np.sign(t) == np.sign(p)))

    # --- Information Coefficient: Pearson correlation ---
    if n_valid < 2 or np.std(t) == 0.0 or np.std(p) == 0.0:
        ic = float("nan")
    else:
        ic_val, _ = stats.pearsonr(t, p)
        ic = float(ic_val)

    logger.debug(
        "Metrics (n={}): RMSE={:.2f}  MAE={:.2f}  R2={:.4f}  DA={:.4f}  IC={:.4f}",
        n_valid,
        rmse,
        mae,
        r2,
        direction_accuracy,
        ic,
    )
    return dict(rmse=rmse, mae=mae, r2=r2, direction_accuracy=direction_accuracy, ic=ic)
