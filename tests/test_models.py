"""Unit tests for energy_trading.models metrics, train, and walk_forward.

All tests use synthetic DataFrames.  No disk I/O or network calls.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from energy_trading.models.metrics import compute_metrics
from energy_trading.models.train import build_models, get_feature_importance
from energy_trading.models.walk_forward import (
    WalkForwardCV,
    prepare_hub_data,
    run_walk_forward,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# Minimal feature set for model tests.  Avoids cross-hub columns that require
# multiple hubs in the same DataFrame.
FEATURE_COLS = [
    "hour_utc",
    "dow",
    "month",
    "is_weekend",
    "is_peak",
    "sin_hour",
    "cos_hour",
    "spread_lag_1d",
    "spread_lag_7d",
    "load_mw_lag1h",
    "spread_vol_24h",
]


def _make_features_df(
    n_years: int = 3,
    start_year: int = 2021,
    hubs: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Minimal synthetic features DataFrame spanning n_years full calendar years.

    Schema matches DataPipeline.load_features_dataset():
    Index: interval_start_utc (UTC hourly DatetimeIndex)
    Columns: hub, dam_lmp, rtm_lmp, spread, load_mw, + FEATURE_COLS
    """
    hubs = hubs or ["HB_NORTH"]
    rng = np.random.default_rng(seed)

    idx = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{start_year + n_years - 1}-12-31 23:00",
        freq="h",
        tz="UTC",
    )
    records = []
    for ts in idx:
        for hub in hubs:
            spread_val = float(rng.normal(5.0, 20.0))
            records.append(
                {
                    "hub": hub,
                    "dam_lmp": 40.0 + rng.normal(0, 5),
                    "rtm_lmp": 35.0 + rng.normal(0, 8),
                    "spread": spread_val,
                    "load_mw": 35_000 + rng.normal(0, 2_000),
                    "hour_utc": ts.hour,
                    "dow": ts.dayofweek,
                    "month": ts.month,
                    "is_weekend": int(ts.dayofweek >= 5),
                    "is_peak": int(6 <= ts.hour <= 21 and ts.dayofweek < 5),
                    "sin_hour": float(np.sin(2 * np.pi * ts.hour / 24)),
                    "cos_hour": float(np.cos(2 * np.pi * ts.hour / 24)),
                    "spread_lag_1d": spread_val + float(rng.normal(0, 1)),
                    "spread_lag_7d": spread_val + float(rng.normal(0, 2)),
                    "load_mw_lag1h": 35_000 + rng.normal(0, 2_000),
                    "spread_vol_24h": abs(float(rng.normal(10, 3))),
                }
            )

    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex(
        [ts for ts in idx for _ in hubs], tz="UTC", name="interval_start_utc"
    )
    return df


# ---------------------------------------------------------------------------
# WalkForwardCV
# ---------------------------------------------------------------------------


class TestWalkForwardCV:
    def _idx(self, start_year: int, n_years: int) -> pd.DatetimeIndex:
        return pd.date_range(
            f"{start_year}-01-01",
            f"{start_year + n_years - 1}-12-31 23:00",
            freq="h",
            tz="UTC",
        )

    def test_split_count_two_years(self):
        """1 training year + 1 test year → 1 fold."""
        cv = WalkForwardCV(min_train_years=1)
        assert cv.n_splits(self._idx(2021, 2)) == 1

    def test_split_count_three_years(self):
        """3 years, min_train=1 → 2 folds."""
        cv = WalkForwardCV(min_train_years=1)
        assert cv.n_splits(self._idx(2021, 3)) == 2

    def test_split_count_five_years(self):
        """5 years, min_train=1 → 4 folds."""
        cv = WalkForwardCV(min_train_years=1)
        assert cv.n_splits(self._idx(2021, 5)) == 4

    def test_no_future_leakage(self):
        """Max training timestamp < min test timestamp in every fold."""
        idx = self._idx(2021, 3)
        cv = WalkForwardCV(min_train_years=1)
        for train_idx, test_idx in cv.split(idx):
            assert idx[train_idx].max() < idx[test_idx].min()

    def test_first_fold_year_assignment(self):
        """Fold 1: train on 2021 only, test on 2022."""
        idx = self._idx(2021, 3)
        cv = WalkForwardCV(min_train_years=1)
        first_train, first_test = next(cv.split(idx))
        assert set(idx[first_train].year) == {2021}
        assert set(idx[first_test].year) == {2022}

    def test_second_fold_year_assignment(self):
        """Fold 2: train on 2021-2022, test on 2023."""
        idx = self._idx(2021, 3)
        cv = WalkForwardCV(min_train_years=1)
        folds = list(cv.split(idx))
        second_train, second_test = folds[1]
        assert set(idx[second_train].year) == {2021, 2022}
        assert set(idx[second_test].year) == {2023}

    def test_indices_are_integer_arrays(self):
        idx = self._idx(2021, 2)
        cv = WalkForwardCV(min_train_years=1)
        for train_idx, test_idx in cv.split(idx):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)
            assert np.issubdtype(train_idx.dtype, np.integer)
            assert np.issubdtype(test_idx.dtype, np.integer)

    def test_raises_with_only_one_year(self):
        idx = self._idx(2021, 1)
        cv = WalkForwardCV(min_train_years=1)
        with pytest.raises(ValueError, match="Not enough years"):
            list(cv.split(idx))

    def test_min_train_years_two(self):
        """min_train_years=2 with 3 years → 1 fold (test on year 3)."""
        idx = self._idx(2021, 3)
        cv = WalkForwardCV(min_train_years=2)
        assert cv.n_splits(idx) == 1

    def test_invalid_min_train_years_raises(self):
        with pytest.raises(ValueError, match="min_train_years"):
            WalkForwardCV(min_train_years=0)


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_returns_required_keys(self):
        y = np.array([1.0, -2.0, 3.0, -1.0, 2.0])
        p = np.array([0.5, -1.0, 2.5, 0.5, 1.5])
        result = compute_metrics(y, p)
        assert set(result.keys()) == {"rmse", "mae", "r2", "direction_accuracy", "ic"}

    def test_perfect_prediction(self):
        y = np.array([1.0, -2.0, 3.0, -1.5, 4.0])
        result = compute_metrics(y, y.copy())
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)
        assert result["mae"] == pytest.approx(0.0, abs=1e-10)
        assert result["r2"] == pytest.approx(1.0, abs=1e-10)
        assert result["direction_accuracy"] == pytest.approx(1.0, abs=1e-10)
        assert result["ic"] == pytest.approx(1.0, abs=1e-10)

    def test_direction_accuracy_random_near_half(self):
        """Independent random arrays should have direction accuracy near 0.5."""
        rng = np.random.default_rng(7)
        y_true = rng.normal(0, 10, size=5_000)
        y_pred = rng.normal(0, 10, size=5_000)
        result = compute_metrics(y_true, y_pred)
        assert 0.40 < result["direction_accuracy"] < 0.60

    def test_all_nan_returns_nan_dict(self):
        result = compute_metrics(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
        assert all(np.isnan(v) for v in result.values())

    def test_partial_nan_uses_valid_rows_only(self):
        """The NaN row (index 1) should be excluded; perfect on valid rows."""
        y_true = np.array([1.0, np.nan, 3.0])
        y_pred = np.array([1.0, 99.0, 3.0])  # 99.0 should be masked
        result = compute_metrics(y_true, y_pred)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_prediction_gives_nan_ic(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([5.0, 5.0, 5.0, 5.0])
        result = compute_metrics(y_true, y_pred)
        assert np.isnan(result["ic"])

    def test_negative_r2_when_worse_than_mean(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # anti-correlated
        result = compute_metrics(y_true, y_pred)
        assert result["r2"] < 0


# ---------------------------------------------------------------------------
# build_models
# ---------------------------------------------------------------------------


class TestBuildModels:
    def test_returns_correct_keys(self):
        assert set(build_models().keys()) == {"linear", "ridge", "lasso", "lgbm"}

    def test_all_have_fit_and_predict(self):
        for name, model in build_models().items():
            assert hasattr(model, "fit"), f"{name} missing .fit"
            assert hasattr(model, "predict"), f"{name} missing .predict"

    def test_linear_models_are_pipelines(self):
        models = build_models()
        for name in ("linear", "ridge", "lasso"):
            assert isinstance(models[name], Pipeline), f"{name} should be Pipeline"

    def test_lgbm_is_lgbm_regressor(self):
        assert isinstance(build_models()["lgbm"], LGBMRegressor)

    def test_lgbm_params_override(self):
        models = build_models(lgbm_params={"n_estimators": 42})
        assert models["lgbm"].n_estimators == 42

    def test_ridge_alpha_applied(self):
        models = build_models(ridge_alpha=10.0)
        assert models["ridge"].named_steps["model"].alpha == 10.0

    def test_lasso_alpha_applied(self):
        models = build_models(lasso_alpha=0.05)
        assert models["lasso"].named_steps["model"].alpha == 0.05


# ---------------------------------------------------------------------------
# get_feature_importance
# ---------------------------------------------------------------------------


class TestGetFeatureImportance:
    _cols = ["f1", "f2", "f3", "f4"]

    def _fit_linear(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, len(self._cols))), columns=self._cols)
        y = rng.normal(size=100)
        model = build_models()["linear"]
        model.fit(X, y)
        return model

    def _fit_lgbm(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(100, len(self._cols))), columns=self._cols)
        y = rng.normal(size=100)
        model = build_models(lgbm_params={"n_estimators": 20})["lgbm"]
        model.fit(X, y)
        return model

    def test_linear_returns_series(self):
        result = get_feature_importance(self._fit_linear(), self._cols)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(self._cols)

    def test_linear_sorted_descending(self):
        result = get_feature_importance(self._fit_linear(), self._cols)
        vals = result.values.tolist()
        assert vals == sorted(vals, reverse=True)

    def test_lgbm_returns_series(self):
        result = get_feature_importance(self._fit_lgbm(), self._cols)
        assert isinstance(result, pd.Series)
        assert set(result.index) == set(self._cols)

    def test_lgbm_sorted_descending(self):
        result = get_feature_importance(self._fit_lgbm(), self._cols)
        vals = result.values.tolist()
        assert vals == sorted(vals, reverse=True)

    def test_unfitted_pipeline_raises(self):
        unfitted = clone(build_models()["linear"])
        with pytest.raises(ValueError, match="fitted"):
            get_feature_importance(unfitted, self._cols)

    def test_wrong_type_raises(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit([[1], [2], [3]])
        with pytest.raises(ValueError, match="Unrecognised model type"):
            get_feature_importance(scaler, ["a"])


# ---------------------------------------------------------------------------
# prepare_hub_data
# ---------------------------------------------------------------------------


class TestPrepareHubData:
    @pytest.fixture()
    def df(self):
        return _make_features_df(n_years=2, start_year=2021, hubs=["HB_NORTH"])

    def test_returns_x_y_tuple(self, df):
        X, y = prepare_hub_data(df, hub="HB_NORTH", feature_cols=FEATURE_COLS)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_x_has_correct_columns(self, df):
        X, _ = prepare_hub_data(df, hub="HB_NORTH", feature_cols=FEATURE_COLS)
        assert list(X.columns) == FEATURE_COLS

    def test_x_y_share_index(self, df):
        X, y = prepare_hub_data(df, hub="HB_NORTH", feature_cols=FEATURE_COLS)
        pd.testing.assert_index_equal(X.index, y.index)

    def test_warmup_rows_removed(self, df):
        X, _ = prepare_hub_data(
            df, hub="HB_NORTH", feature_cols=FEATURE_COLS, min_lag_hours=168
        )
        cutoff = df.index.min() + pd.Timedelta(hours=168)
        assert X.index.min() >= cutoff

    def test_no_nan_in_output(self, df):
        X, y = prepare_hub_data(df, hub="HB_NORTH", feature_cols=FEATURE_COLS)
        assert not X.isna().any().any()
        assert not y.isna().any()

    def test_raises_on_unknown_hub(self, df):
        with pytest.raises(ValueError, match="HB_FAKE"):
            prepare_hub_data(df, hub="HB_FAKE", feature_cols=FEATURE_COLS)

    def test_index_name_preserved(self, df):
        X, _ = prepare_hub_data(df, hub="HB_NORTH", feature_cols=FEATURE_COLS)
        assert X.index.name == "interval_start_utc"


# ---------------------------------------------------------------------------
# run_walk_forward
# ---------------------------------------------------------------------------


class TestRunWalkForward:
    @pytest.fixture()
    def df_3yr(self):
        """3-year synthetic dataset for one hub → 2 folds."""
        return _make_features_df(n_years=3, start_year=2021, hubs=["HB_NORTH"])

    @pytest.fixture()
    def fast_models(self):
        """Fast minimal models for integration tests (LightGBM with 10 trees)."""
        return build_models(lgbm_params={"n_estimators": 10})

    def test_returns_two_dataframes(self, df_3yr, fast_models):
        metrics_df, oos_df = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        assert isinstance(metrics_df, pd.DataFrame)
        assert isinstance(oos_df, pd.DataFrame)

    def test_metrics_df_required_columns(self, df_3yr, fast_models):
        metrics_df, _ = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        required = {
            "fold", "test_year", "train_years_str", "hub", "model",
            "n_train", "n_test", "rmse", "mae", "r2", "direction_accuracy", "ic",
        }
        assert required.issubset(set(metrics_df.columns))

    def test_metrics_df_row_count(self, df_3yr, fast_models):
        """3 years, min_train=1 → 2 folds × 4 models = 8 rows."""
        metrics_df, _ = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        assert len(metrics_df) == 2 * len(fast_models)

    def test_oos_has_spread_pred_columns(self, df_3yr, fast_models):
        _, oos_df = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        for name in fast_models:
            assert f"spread_pred_{name}" in oos_df.columns

    def test_oos_index_is_datetimeindex(self, df_3yr, fast_models):
        _, oos_df = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        assert isinstance(oos_df.index, pd.DatetimeIndex)
        assert oos_df.index.name == "interval_start_utc"

    def test_training_year_absent_from_oos(self, df_3yr, fast_models):
        """Year 2021 is training-only and must not appear in OOS output."""
        _, oos_df = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        assert 2021 not in oos_df.index.year

    def test_oos_only_contains_test_years(self, df_3yr, fast_models):
        _, oos_df = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        assert set(oos_df.index.year).issubset({2022, 2023})

    def test_hub_column_correct(self, df_3yr, fast_models):
        _, oos_df = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        assert "hub" in oos_df.columns
        assert (oos_df["hub"] == "HB_NORTH").all()

    def test_metrics_hub_column_correct(self, df_3yr, fast_models):
        metrics_df, _ = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        assert (metrics_df["hub"] == "HB_NORTH").all()

    def test_numeric_metrics_are_finite(self, df_3yr, fast_models):
        metrics_df, _ = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        for col in ("rmse", "mae", "direction_accuracy"):
            assert metrics_df[col].notna().all(), f"{col} has NaN"
            assert np.isfinite(metrics_df[col].values).all(), f"{col} has Inf"

    def test_oos_predictions_are_float(self, df_3yr, fast_models):
        _, oos_df = run_walk_forward(
            df_3yr, hub="HB_NORTH", feature_cols=FEATURE_COLS, models=fast_models
        )
        for name in fast_models:
            col = f"spread_pred_{name}"
            assert pd.api.types.is_float_dtype(oos_df[col]), f"{col} not float"
            assert oos_df[col].notna().all(), f"{col} has NaN"
