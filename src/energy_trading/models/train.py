"""Model factory and feature importance extraction for spread forecasting.

This module creates *unfitted* estimator objects only.  Callers (e.g.
``walk_forward.run_walk_forward``) are responsible for cloning and fitting
on each training fold, ensuring no state bleeds between folds.

Model catalogue
---------------
linear   OLS via ``LinearRegression``, wrapped in a ``StandardScaler`` Pipeline.
ridge    Ridge regression (L2) with ``StandardScaler`` Pipeline.
lasso    Lasso regression (L1) with ``StandardScaler`` Pipeline.
lgbm     ``LGBMRegressor`` no scaler needed (tree models are scale-invariant).

All four expose the standard sklearn ``.fit(X, y)`` / ``.predict(X)`` interface,
so they can be used interchangeably in the walk-forward loop.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor
from loguru import logger
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(
    ridge_alpha: float = 1.0,
    lasso_alpha: float = 0.1,
    lgbm_params: dict | None = None,
) -> dict[str, Pipeline | LGBMRegressor]:
    """Construct a dictionary of unfitted estimators ready for walk-forward CV.

    Linear models are wrapped in a ``StandardScaler`` Pipeline because
    spread features span vastly different scales (MW in the thousands for
    load features vs. fractions for cyclical encodings).  LightGBM is used
    directly gradient-boosted trees are scale-invariant.

    Args:
        ridge_alpha: L2 regularisation strength for Ridge.  Larger values
            increase regularisation.  Default: 1.0.
        lasso_alpha: L1 regularisation strength for Lasso.  Default: 0.1.
        lgbm_params: Optional dict of ``LGBMRegressor`` kwargs merged over
            the conservative defaults.  The caller can override any
            hyperparameter without touching this function.

    Returns:
        Dict with keys ``"linear"``, ``"ridge"``, ``"lasso"``, ``"lgbm"``,
        each mapping to an *unfitted* estimator.
    """
    _lgbm_defaults: dict = dict(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        importance_type="gain",
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    if lgbm_params:
        _lgbm_defaults.update(lgbm_params)

    def _pipeline(estimator) -> Pipeline:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])

    models: dict[str, Pipeline | LGBMRegressor] = {
        "linear": _pipeline(LinearRegression()),
        "ridge": _pipeline(Ridge(alpha=ridge_alpha)),
        "lasso": _pipeline(Lasso(alpha=lasso_alpha, max_iter=5_000)),
        "lgbm": LGBMRegressor(**_lgbm_defaults),
    }

    logger.info(
        "Built {} unfitted models: {} (ridge_alpha={}, lasso_alpha={})",
        len(models),
        list(models.keys()),
        ridge_alpha,
        lasso_alpha,
    )
    return models


def get_feature_importance(
    model: Pipeline | LGBMRegressor,
    feature_cols: list[str],
) -> pd.Series:
    """Extract feature importances from a *fitted* estimator.

    For linear models (sklearn ``Pipeline`` containing a linear estimator at
    step ``"model"``), the absolute coefficient values are used as importance
    scores so that large negative and large positive coefficients both rank
    highly.  For LightGBM, ``feature_importances_`` (gain-based, as set by
    ``importance_type="gain"`` at construction) is used.

    Args:
        model: A *fitted* estimator either a sklearn ``Pipeline`` built by
            ``build_models()`` or a fitted ``LGBMRegressor``.
        feature_cols: Ordered list of feature names matching the training
            columns.  Must have the same length as the estimator's coefficient
            or importance array.

    Returns:
        ``pd.Series`` indexed by feature name, values are non-negative
        importance scores, sorted descending.

    Raises:
        ValueError: If the estimator type is not recognised, the model has
            not been fitted, or ``len(feature_cols)`` does not match the
            number of importances.
    """
    if isinstance(model, Pipeline):
        step = model.named_steps.get("model")
        if step is None or not hasattr(step, "coef_"):
            raise ValueError(
                "Pipeline does not contain a fitted linear model at step 'model'. "
                "Ensure build_models() was used and the model has been fitted."
            )
        importances = pd.Series(
            abs(step.coef_),
            index=feature_cols,
            name="importance",
        )
    elif isinstance(model, LGBMRegressor):
        if not hasattr(model, "feature_importances_"):
            raise ValueError("LGBMRegressor has not been fitted yet.")
        importances = pd.Series(
            model.feature_importances_.astype(float),
            index=feature_cols,
            name="importance",
        )
    else:
        raise ValueError(
            f"Unrecognised model type: {type(model).__name__}. "
            "Expected a Pipeline (linear model) or LGBMRegressor."
        )

    return importances.sort_values(ascending=False)
