"""Expanding-window walk-forward cross-validation for spread forecasting.

Walk-forward scheme
-------------------
Given data spanning years [y0, y1, ..., yn] with ``min_train_years=1``:

  Fold 1: train=[y0],          test=[y1]
  Fold 2: train=[y0, y1],      test=[y2]
  ...
  Fold k: train=[y0..y(k-1)],  test=[yk]

This guarantees strict temporal ordering: every test observation is strictly
after every training observation, with zero information leakage.

Per-hub modelling
-----------------
``run_walk_forward`` operates on one hub at a time.  The caller iterates
over hubs and concatenates results.  This keeps each hub's model independent
and enables interpretable per-hub feature importance analysis.
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import clone

from energy_trading.features.engineering import drop_warmup_rows
from energy_trading.models.metrics import compute_metrics


class WalkForwardCV:
    """Expanding-window, calendar-year walk-forward cross-validator.

    Yields ``(train_indices, test_indices)`` tuples where indices are
    1-D integer arrays of positions into the input DatetimeIndex, compatible
    with ``.iloc[]``.

    Args:
        min_train_years: Minimum number of full calendar years required in
            the training window before the first test fold.  Default: 1.

    Raises:
        ValueError: If ``min_train_years < 1``.
    """

    def __init__(self, min_train_years: int = 1) -> None:
        if min_train_years < 1:
            raise ValueError(
                f"min_train_years must be >= 1, got {min_train_years}."
            )
        self.min_train_years = min_train_years

    def split(
        self,
        index: pd.DatetimeIndex,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(train_indices, test_indices)`` for each expanding fold.

        The training set for fold *k* contains all timestamps from every
        year strictly before the test year.  The test set contains all
        timestamps in the test year.

        Args:
            index: UTC hourly DatetimeIndex of the full dataset (for one hub,
                post-warmup-drop).

        Yields:
            Tuples of ``(train_idx, test_idx)`` 1-D ``np.ndarray`` of
            integer positions suitable for ``.iloc[]``.

        Raises:
            ValueError: If there are fewer than ``min_train_years + 1``
                distinct calendar years in ``index``.
        """
        years = sorted(index.year.unique())
        first_test_year = years[0] + self.min_train_years

        if first_test_year > years[-1]:
            raise ValueError(
                f"Not enough years in index to form even one fold. "
                f"Have years {years}, need at least "
                f"{self.min_train_years + 1} distinct years "
                f"(min_train_years={self.min_train_years})."
            )

        for test_year in years:
            if test_year < first_test_year:
                continue

            train_idx = np.where(index.year < test_year)[0]
            test_idx = np.where(index.year == test_year)[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def n_splits(self, index: pd.DatetimeIndex) -> int:
        """Return the number of folds for the given index.

        Args:
            index: Same DatetimeIndex that will be passed to ``split()``.

        Returns:
            Total number of ``(train, test)`` fold pairs.
        """
        return sum(1 for _ in self.split(index))

    def get_fold_metadata(self, index: pd.DatetimeIndex) -> list[dict]:
        """Return lightweight metadata for each fold (no index arrays).

        Useful for logging and constructing the metrics DataFrame without
        materialising the full index arrays twice.

        Args:
            index: Same DatetimeIndex as passed to ``split()``.

        Returns:
            List of dicts with keys: ``fold``, ``test_year``,
            ``train_years_str``.
        """
        years = sorted(index.year.unique())
        first_test_year = years[0] + self.min_train_years
        metadata = []
        fold_num = 1
        for test_year in years:
            if test_year < first_test_year:
                continue
            train_years = [y for y in years if y < test_year]
            if len(train_years) > 1:
                years_str = f"{train_years[0]}-{train_years[-1]}"
            else:
                years_str = str(train_years[0])
            metadata.append(
                dict(fold=fold_num, test_year=test_year, train_years_str=years_str)
            )
            fold_num += 1
        return metadata


def prepare_hub_data(
    features_df: pd.DataFrame,
    hub: str,
    feature_cols: list[str],
    target_col: str = "spread",
    min_lag_hours: int = 168,
) -> tuple[pd.DataFrame, pd.Series]:
    """Filter, warm-up drop, and NaN-clean data for a single hub.

    Args:
        features_df: Full feature matrix from
            ``DataPipeline.load_features_dataset()``.  Index:
            ``interval_start_utc`` (UTC hourly).  Must contain a ``"hub"``
            column.
        hub: Hub identifier, e.g. ``"HB_NORTH"``.
        feature_cols: Feature column names to include as ``X``.
        target_col: Target column name.  Default: ``"spread"``.
        min_lag_hours: Warm-up period (hours) to discard from the start of
            this hub's history.  Default: 168 (one week equal to the
            longest lag feature).

    Returns:
        Tuple ``(X, y)`` where:

        * ``X`` is a ``pd.DataFrame`` with columns ``feature_cols`` and a
          UTC DatetimeIndex.
        * ``y`` is a ``pd.Series`` with the same index.

        Both have had NaN rows dropped (single ``dropna`` pass over
        ``feature_cols + [target_col]``).

    Raises:
        ValueError: If ``hub`` is not present in ``features_df["hub"]``.
        ValueError: If ``target_col`` is not a column of ``features_df``.
        ValueError: If any element of ``feature_cols`` is absent from the
            hub-filtered subset.
    """
    if hub not in features_df["hub"].values:
        raise ValueError(
            f"Hub {hub!r} not found in features_df. "
            f"Available hubs: {sorted(features_df['hub'].unique().tolist())}"
        )
    if target_col not in features_df.columns:
        raise ValueError(
            f"Target column {target_col!r} not in features_df columns."
        )

    hub_df = features_df[features_df["hub"] == hub].copy()
    hub_df = drop_warmup_rows(hub_df, min_lag_hours=min_lag_hours)

    missing = [c for c in feature_cols if c not in hub_df.columns]
    if missing:
        raise ValueError(
            f"Feature columns missing from hub subset for {hub!r}: {missing}"
        )

    needed = feature_cols + [target_col]
    clean = hub_df[needed].dropna(subset=needed)

    n_dropped = len(hub_df) - len(clean)
    logger.info(
        "prepare_hub_data(hub={}): {} usable rows (dropped {} NaN rows).",
        hub,
        len(clean),
        n_dropped,
    )

    X = clean[feature_cols]
    y = clean[target_col].rename(target_col)
    return X, y


def run_walk_forward(
    features_df: pd.DataFrame,
    hub: str,
    feature_cols: list[str],
    models: dict[str, Any],
    target_col: str = "spread",
    min_train_years: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run expanding-window walk-forward CV for one hub across all models.

    Each model is **cloned** (via ``sklearn.base.clone``) at the start of
    every ``(fold, model)`` combination so that no fitted state persists
    between folds.

    Args:
        features_df: Full feature matrix.  See ``prepare_hub_data``.
        hub: Hub to model, e.g. ``"HB_NORTH"``.
        feature_cols: Ordered list of feature column names.
        models: Dict of *unfitted* estimators (e.g. from ``build_models()``).
            Keys are model name strings; values are sklearn-compatible
            estimators with ``.fit`` / ``.predict``.
        target_col: Target column name.  Default: ``"spread"``.
        min_train_years: Minimum training years before first test fold.
            Default: 1.

    Returns:
        Tuple ``(metrics_df, oos_df)`` where:

        **metrics_df** one row per ``(fold × model)`` with columns:
          ``fold``, ``test_year``, ``train_years_str``, ``hub``, ``model``,
          ``n_train``, ``n_test``, ``rmse``, ``mae``, ``r2``,
          ``direction_accuracy``, ``ic``.

        **oos_df** one row per test-set timestamp with columns:
          ``hub``, ``spread`` (actual), ``spread_pred_{model}`` for each
          model.  Index is ``interval_start_utc`` (UTC), sorted ascending.
          Test-set timestamps are mutually exclusive across folds, so no
          row collisions occur.
    """
    X, y = prepare_hub_data(
        features_df,
        hub=hub,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    cv = WalkForwardCV(min_train_years=min_train_years)
    fold_metadata = cv.get_fold_metadata(X.index)
    n_folds = len(fold_metadata)

    metrics_rows: list[dict] = []
    # Accumulate OOS predictions keyed by timestamp (folds never overlap).
    oos_records: dict[pd.Timestamp, dict] = {}

    for fold_info, (train_idx, test_idx) in zip(
        fold_metadata, cv.split(X.index)
    ):
        fold = fold_info["fold"]
        test_year = fold_info["test_year"]
        train_years_str = fold_info["train_years_str"]

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        logger.info(
            "Fold {}/{}: hub={}, train={}, test={}, n_train={}, n_test={}",
            fold,
            n_folds,
            hub,
            train_years_str,
            test_year,
            len(X_train),
            len(X_test),
        )

        # Seed OOS records with actuals for this fold's test timestamps.
        for ts, actual in zip(X_test.index, y_test.values):
            oos_records[ts] = {"hub": hub, target_col: float(actual)}

        for model_name, model_template in models.items():
            fitted = clone(model_template)
            fitted.fit(X_train, y_train)
            preds = fitted.predict(X_test)

            for ts, pred_val in zip(X_test.index, preds):
                oos_records[ts][f"spread_pred_{model_name}"] = float(pred_val)

            fold_metrics = compute_metrics(y_test.values, preds)
            metrics_rows.append(
                dict(
                    fold=fold,
                    test_year=test_year,
                    train_years_str=train_years_str,
                    hub=hub,
                    model=model_name,
                    n_train=len(X_train),
                    n_test=len(X_test),
                    **fold_metrics,
                )
            )

    metrics_df = pd.DataFrame(metrics_rows)

    oos_df = pd.DataFrame.from_dict(oos_records, orient="index")
    oos_df.index.name = "interval_start_utc"
    oos_df = oos_df.sort_index()

    logger.info(
        "run_walk_forward complete: hub={}, {} folds, {} OOS rows.",
        hub,
        n_folds,
        len(oos_df),
    )
    return metrics_df, oos_df
