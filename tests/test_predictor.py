"""
在线预测模块 — 单元测试

测试模型加载、预测函数逻辑、输入处理。
"""

import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from src.predictor import (
    predict,
    FEATURE_DEFAULTS,
    JOB_OPTIONS,
    MARITAL_OPTIONS,
    EDUCATION_OPTIONS,
    MONTH_OPTIONS,
    DAY_OF_WEEK_OPTIONS,
    POUTCOME_OPTIONS,
)
from src.data_loader import get_prediction_features


@pytest.fixture
def sample_model():
    """Create and train a minimal model for prediction testing."""
    np.random.seed(42)
    n = 500

    # Create synthetic training data
    df = pd.DataFrame(
        {
            "age": np.random.randint(18, 95, n),
            "job": np.random.choice(JOB_OPTIONS, n),
            "marital": np.random.choice(MARITAL_OPTIONS, n),
            "education": np.random.choice(EDUCATION_OPTIONS, n),
            "default": np.random.choice(["no", "yes", "unknown"], n, p=[0.8, 0.1, 0.1]),
            "housing": np.random.choice(["no", "yes", "unknown"], n, p=[0.3, 0.6, 0.1]),
            "loan": np.random.choice(["no", "yes", "unknown"], n, p=[0.7, 0.2, 0.1]),
            "contact": np.random.choice(["cellular", "telephone"], n),
            "month": np.random.choice(MONTH_OPTIONS, n),
            "day_of_week": np.random.choice(DAY_OF_WEEK_OPTIONS, n),
            "campaign": np.random.randint(1, 20, n),
            "pdays": np.random.choice([999, 3, 7, 14], n),
            "previous": np.random.randint(0, 5, n),
            "poutcome": np.random.choice(POUTCOME_OPTIONS, n),
            "emp_var_rate": np.random.uniform(-3.5, 1.5, n),
            "cons_price_index": np.random.uniform(92, 95, n),
            "cons_conf_index": np.random.uniform(-51, -35, n),
            "lending_rate3m": np.random.uniform(0.6, 5.0, n),
            "nr_employed": np.random.uniform(4900, 5200, n),
        }
    )
    y = np.random.choice([0, 1], n, p=[0.85, 0.15])

    # Build a simple pipeline
    num_feats = [
        "age", "campaign", "pdays", "previous", "emp_var_rate",
        "cons_price_index", "cons_conf_index", "lending_rate3m", "nr_employed",
    ]
    cat_feats = [
        "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "poutcome",
    ]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_feats),
        ("cat", cat_pipe, cat_feats),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    pipeline.fit(df, y)

    return pipeline, list(df.columns)


class TestPredictionFunction:
    """Tests for the predict function."""

    def test_predict_returns_label_and_probability(self, sample_model):
        """Should return a string label and a float probability."""
        model, feature_names = sample_model
        user_input = {
            "age": 40,
            "job": "admin.",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp_var_rate": 1.4,
            "cons_price_index": 93.0,
            "cons_conf_index": -40.0,
            "lending_rate3m": 1.5,
            "nr_employed": 5100.0,
        }

        label, prob = predict(model, None, user_input, feature_names)

        assert isinstance(label, str)
        assert isinstance(prob, float)
        assert 0 <= prob <= 1
        assert "认购" in label

    def test_predict_positive_label(self, sample_model):
        """Prediction label should indicate yes/no."""
        model, feature_names = sample_model
        user_input = FEATURE_DEFAULTS.copy()

        label, prob = predict(model, None, user_input, feature_names)

        # Either result is valid
        assert label in ("会认购 ✅", "不会认购 ❌")

    def test_predict_with_boundary_values(self, sample_model):
        """Should handle extreme/missing-like values."""
        model, feature_names = sample_model
        user_input = {
            "age": 95,
            "job": "unknown",
            "marital": "unknown",
            "education": "illiterate",
            "default": "yes",
            "housing": "no",
            "loan": "yes",
            "contact": "telephone",
            "month": "dec",
            "day_of_week": "fri",
            "campaign": 50,
            "pdays": 0,
            "previous": 10,
            "poutcome": "success",
            "emp_var_rate": -3.5,
            "cons_price_index": 85.0,
            "cons_conf_index": -60.0,
            "lending_rate3m": 10.0,
            "nr_employed": 4500.0,
        }

        label, prob = predict(model, None, user_input, feature_names)

        assert 0 <= prob <= 1
        assert isinstance(label, str)

    def test_predict_consistent_output(self, sample_model):
        """Same input should produce same output."""
        model, feature_names = sample_model
        user_input = FEATURE_DEFAULTS.copy()

        label1, prob1 = predict(model, None, user_input, feature_names)
        label2, prob2 = predict(model, None, user_input, feature_names)

        assert label1 == label2
        assert prob1 == pytest.approx(prob2)


class TestFeatureDefaults:
    """Tests for feature definitions and defaults."""

    def test_defaults_cover_all_prediction_features(self):
        """FEATURE_DEFAULTS should have entries for all prediction features."""
        pred_features = get_prediction_features()
        for feat in pred_features:
            assert feat in FEATURE_DEFAULTS, f"Missing default for '{feat}'"

    def test_job_options_not_empty(self):
        """Job options should be a non-empty list."""
        assert len(JOB_OPTIONS) > 0
        assert "admin." in JOB_OPTIONS

    def test_month_options_cover_all(self):
        """Month options should have 12 entries."""
        assert len(MONTH_OPTIONS) == 12

    def test_day_of_week_options(self):
        """Day of week should have 5 entries."""
        assert len(DAY_OF_WEEK_OPTIONS) == 5

    def test_poutcome_options(self):
        """Poutcome options should include nonexistent."""
        assert "nonexistent" in POUTCOME_OPTIONS


class TestModelArtifact:
    """Tests for model save/load compatibility."""

    def test_save_and_load_for_prediction(self, sample_model):
        """Prediction should work after save→load cycle."""
        model, feature_names = sample_model

        artifact = {
            "pipeline": model,
            "feature_names": feature_names,
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = f.name

        try:
            with open(tmp_path, "wb") as f:
                pickle.dump(artifact, f)

            with open(tmp_path, "rb") as f:
                loaded = pickle.load(f)

            user_input = FEATURE_DEFAULTS.copy()
            label, prob = predict(
                loaded["pipeline"], None, user_input, loaded["feature_names"]
            )

            assert isinstance(label, str)
            assert 0 <= prob <= 1
        finally:
            os.unlink(tmp_path)

    def test_artifact_has_feature_names(self, sample_model):
        """Saved artifact should include feature names."""
        model, feature_names = sample_model
        artifact = {"pipeline": model, "feature_names": feature_names}
        assert "feature_names" in artifact
        assert len(artifact["feature_names"]) == 19  # excludes duration
