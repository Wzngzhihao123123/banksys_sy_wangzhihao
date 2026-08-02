"""
模型训练模块 — 单元测试

测试特征预处理、Pipeline 构建、训练流程、评估、保存/加载、质量门禁。
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model_trainer import (
    build_preprocessor,
    prepare_data,
    build_model_pipeline,
    train_and_evaluate,
    evaluate_model,
    save_model,
    check_model_quality,
    AUC_THRESHOLD,
    ACCURACY_THRESHOLD,
)
from src.data_loader import (
    get_prediction_features,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


@pytest.fixture
def sample_df():
    """Create a sample DataFrame matching the bank marketing schema."""
    np.random.seed(42)
    n = 500
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "age": np.random.randint(18, 95, n),
            "job": np.random.choice(
                ["admin.", "blue-collar", "technician", "services", "management"], n
            ),
            "marital": np.random.choice(["married", "single", "divorced"], n),
            "education": np.random.choice(
                ["university.degree", "high.school", "professional.course", "basic.9y"],
                n,
            ),
            "default": np.random.choice(["no", "yes", "unknown"], n, p=[0.8, 0.1, 0.1]),
            "housing": np.random.choice(["no", "yes", "unknown"], n, p=[0.3, 0.6, 0.1]),
            "loan": np.random.choice(["no", "yes", "unknown"], n, p=[0.7, 0.2, 0.1]),
            "contact": np.random.choice(["cellular", "telephone"], n),
            "month": np.random.choice(
                ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], n
            ),
            "day_of_week": np.random.choice(["mon", "tue", "wed", "thu", "fri"], n),
            "duration": np.random.randint(0, 3000, n),
            "campaign": np.random.randint(1, 20, n),
            "pdays": np.random.choice([999, 3, 7, 14], n),
            "previous": np.random.randint(0, 5, n),
            "poutcome": np.random.choice(["failure", "nonexistent", "success"], n),
            "emp_var_rate": np.random.uniform(-3.5, 1.5, n),
            "cons_price_index": np.random.uniform(92, 95, n),
            "cons_conf_index": np.random.uniform(-51, -35, n),
            "lending_rate3m": np.random.uniform(0.6, 5.0, n),
            "nr_employed": np.random.uniform(4900, 5200, n),
            "subscribe": np.random.choice(["no", "yes"], n, p=[0.85, 0.15]),
        }
    )


class TestBuildPreprocessor:
    """Tests for the preprocessor construction."""

    def test_build_preprocessor_returns_transformer(self):
        """Should return a ColumnTransformer."""
        num_feats = ["age", "duration", "campaign"]
        cat_feats = ["job", "marital"]
        preprocessor = build_preprocessor(num_feats, cat_feats)
        assert preprocessor is not None
        assert hasattr(preprocessor, "fit")
        assert hasattr(preprocessor, "transform")

    def test_preprocessor_output_shape(self, sample_df):
        """Preprocessor should output correct shape after fit/transform."""
        num_feats = ["age", "campaign", "previous"]
        cat_feats = ["job", "marital", "education"]
        X = sample_df[num_feats + cat_feats]

        preprocessor = build_preprocessor(num_feats, cat_feats)
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed.shape[0] == len(sample_df)
        # One-hot encoding expands categorical features
        assert X_transformed.shape[1] > len(num_feats)

    def test_preprocessor_handles_missing_values(self, sample_df):
        """Preprocessor should handle NaN values without error."""
        df = sample_df.copy()
        df.loc[0:5, "age"] = None
        df.loc[10:15, "job"] = None

        num_feats = ["age", "campaign"]
        cat_feats = ["job", "marital"]

        preprocessor = build_preprocessor(num_feats, cat_feats)
        X_transformed = preprocessor.fit_transform(df[num_feats + cat_feats])

        assert not np.any(np.isnan(X_transformed))
        assert X_transformed.shape[0] == len(df)

    def test_preprocessor_handles_unknown_category(self, sample_df):
        """Preprocessor should handle unseen categories gracefully."""
        num_feats = ["age"]
        cat_feats = ["job"]

        X_train = sample_df.iloc[:400][num_feats + cat_feats]
        X_test = sample_df.iloc[400:][num_feats + cat_feats].copy()
        X_test["job"] = "unknown_new_category"

        preprocessor = build_preprocessor(num_feats, cat_feats)
        preprocessor.fit(X_train)
        X_transformed = preprocessor.transform(X_test)

        assert X_transformed.shape[0] == len(X_test)


class TestPrepareData:
    """Tests for data preparation."""

    def test_prepare_data_excludes_duration(self, sample_df):
        """Features should exclude 'duration' and 'id'."""
        X, y, num_feats, cat_feats = prepare_data(sample_df)
        assert "duration" not in X.columns
        assert "id" not in X.columns

    def test_prepare_data_target_is_binary(self, sample_df):
        """Target y should be 0/1 integers."""
        _, y, _, _ = prepare_data(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_prepare_data_no_nan_in_target(self, sample_df):
        """Target should have no NaN values."""
        _, y, _, _ = prepare_data(sample_df)
        assert not y.isnull().any()

    def test_prepare_data_positive_rate_reasonable(self, sample_df):
        """Positive rate should be between 5% and 50%."""
        _, y, _, _ = prepare_data(sample_df)
        rate = y.mean()
        assert 0.05 < rate < 0.5


class TestModelPipeline:
    """Tests for the full model pipeline."""

    def test_build_model_pipeline_is_pipeline(self, sample_df):
        """Should return a sklearn Pipeline."""
        num_feats = ["age", "campaign"]
        cat_feats = ["job"]
        pipeline = build_model_pipeline(num_feats, cat_feats)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_fit_predict(self, sample_df):
        """Pipeline should fit and predict without error."""
        X, y, num_feats, cat_feats = prepare_data(sample_df)
        pipeline = build_model_pipeline(num_feats, cat_feats)

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)

        assert len(y_pred) == len(y)
        assert y_proba.shape[1] == 2
        assert set(y_pred).issubset({0, 1})


class TestTrainAndEvaluate:
    """Tests for the training and evaluation function."""

    def test_returns_all_keys(self, sample_df):
        """Should return expected result keys."""
        from sklearn.model_selection import train_test_split

        X, y, num_feats, cat_feats = prepare_data(sample_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = train_and_evaluate(
            X_train, y_train, X_test, y_test, num_feats, cat_feats
        )

        assert "best_pipeline" in results
        assert "best_model_name" in results
        assert "best_auc" in results
        assert "all_results" in results

    def test_best_auc_is_valid(self, sample_df):
        """Best AUC should be a valid float between 0 and 1."""
        from sklearn.model_selection import train_test_split

        X, y, num_feats, cat_feats = prepare_data(sample_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = train_and_evaluate(
            X_train, y_train, X_test, y_test, num_feats, cat_feats
        )

        assert 0 <= results["best_auc"] <= 1

    def test_both_models_trained(self, sample_df):
        """Should train both LogisticRegression and RandomForest."""
        from sklearn.model_selection import train_test_split

        X, y, num_feats, cat_feats = prepare_data(sample_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        results = train_and_evaluate(
            X_train, y_train, X_test, y_test, num_feats, cat_feats
        )

        assert "LogisticRegression" in results["all_results"]
        assert "RandomForest" in results["all_results"]


class TestEvaluateModel:
    """Tests for the evaluate_model function."""

    def test_returns_all_metrics(self, sample_df):
        """Should return accuracy, precision, recall, f1, auc."""
        from sklearn.model_selection import train_test_split

        X, y, num_feats, cat_feats = prepare_data(sample_df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        pipeline = build_model_pipeline(num_feats, cat_feats)
        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)

        for key in ["accuracy", "precision", "recall", "f1", "auc"]:
            assert key in metrics
            assert 0 <= metrics[key] <= 1


class TestSaveModel:
    """Tests for model persistence."""

    def test_save_and_load_model(self, sample_df):
        """Saved model should be loadable."""
        X, y, num_feats, cat_feats = prepare_data(sample_df)
        pipeline = build_model_pipeline(num_feats, cat_feats)
        pipeline.fit(X, y)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            save_path = save_model(pipeline, list(X.columns), tmp_path)
            assert os.path.exists(save_path)

            with open(save_path, "rb") as f:
                artifact = pickle.load(f)

            assert "pipeline" in artifact
            assert "feature_names" in artifact
            assert isinstance(artifact["pipeline"], Pipeline)

            # Verify it still works
            loaded_pipeline = artifact["pipeline"]
            y_pred = loaded_pipeline.predict(X.head(5))
            assert len(y_pred) == 5
        finally:
            os.unlink(tmp_path)


class TestCheckModelQuality:
    """Tests for quality threshold checking."""

    def test_passes_with_good_metrics(self):
        """Should pass when metrics exceed thresholds."""
        metrics = {"auc": 0.85, "accuracy": 0.88}
        passed, failures = check_model_quality(metrics)
        assert passed is True
        assert len(failures) == 0

    def test_fails_with_low_auc(self):
        """Should fail when AUC is below threshold."""
        metrics = {"auc": 0.55, "accuracy": 0.85}
        passed, failures = check_model_quality(metrics)
        assert passed is False
        assert len(failures) > 0

    def test_fails_with_low_accuracy(self):
        """Should fail when accuracy is below threshold."""
        metrics = {"auc": 0.85, "accuracy": 0.60}
        passed, failures = check_model_quality(metrics)
        assert passed is False

    def test_failure_messages_are_strings(self):
        """Failure messages should be human-readable strings."""
        metrics = {"auc": 0.50, "accuracy": 0.50}
        _, failures = check_model_quality(metrics)
        for msg in failures:
            assert isinstance(msg, str)
            assert len(msg) > 0
