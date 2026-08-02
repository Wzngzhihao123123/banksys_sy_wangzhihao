"""
数据加载与预处理模块 — 单元测试
"""

import os
import tempfile

import pandas as pd
import numpy as np
import pytest

from src.data_loader import (
    load_data,
    load_test_data,
    get_data_summary,
    get_numerical_features,
    get_categorical_features,
    get_prediction_features,
    get_target_name,
    get_missing_report,
    get_feature_stats,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    PREDICTION_FEATURES,
)


@pytest.fixture
def sample_df():
    """Create a minimal sample DataFrame matching the real schema."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "id": range(1, n + 1),
            "age": np.random.randint(18, 95, n),
            "job": np.random.choice(
                ["admin.", "blue-collar", "technician", "services", "management"], n
            ),
            "marital": np.random.choice(["married", "single", "divorced"], n),
            "education": np.random.choice(
                ["university.degree", "high.school", "professional.course", "basic.9y"], n
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


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_data_file_not_found(self):
        """Should raise FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_data("/nonexistent/path.csv")

    def test_load_data_from_temp_file(self, sample_df):
        """Should load data from a CSV file successfully."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            sample_df.to_csv(f, index=False)
            tmp_path = f.name

        try:
            df = load_data(tmp_path)
            assert len(df) == 200
            assert "subscribe" in df.columns
        finally:
            os.unlink(tmp_path)

    def test_load_test_data_not_found(self):
        """Should raise FileNotFoundError for missing test data."""
        with pytest.raises(FileNotFoundError):
            load_test_data("/nonexistent/test.csv")


class TestDataSummary:
    """Tests for summary and statistics functions."""

    def test_get_data_summary_basic(self, sample_df):
        """Should return correct basic counts."""
        summary = get_data_summary(sample_df)
        assert summary["rows"] == 200
        assert summary["columns"] == 22  # id + 20 features + target
        assert summary["missing_total"] == 0
        assert "target_distribution" in summary
        assert "positive_rate" in summary

    def test_get_data_summary_with_missing(self, sample_df):
        """Should detect missing values."""
        df = sample_df.copy()
        df.loc[0:4, "job"] = None
        summary = get_data_summary(df)
        assert summary["missing_total"] > 0
        assert summary["missing_pct"] > 0

    def test_get_data_summary_target_distribution(self, sample_df):
        """Should report correct target distribution."""
        summary = get_data_summary(sample_df)
        assert "yes" in summary["target_distribution"]
        assert "no" in summary["target_distribution"]
        total = sum(summary["target_distribution"].values())
        assert total == 200

    def test_get_missing_report(self, sample_df):
        """Should generate per-column missing report."""
        df = sample_df.copy()
        df.loc[0:2, "age"] = None
        report = get_missing_report(df)
        assert len(report) == len(df.columns)
        age_row = report[report["feature"] == "age"]
        assert age_row["missing_count"].values[0] == 3

    def test_get_feature_stats(self, sample_df):
        """Should return numerical and categorical statistics."""
        num_stats, cat_stats = get_feature_stats(sample_df)
        assert not num_stats.empty
        assert not cat_stats.empty
        # Numerical stats should have count, mean, std, etc.
        assert "mean" in num_stats.index


class TestFeatureDefinitions:
    """Tests for feature list functions."""

    def test_numerical_features_not_empty(self):
        """Should return non-empty list."""
        feats = get_numerical_features()
        assert len(feats) > 0
        assert "age" in feats

    def test_categorical_features_not_empty(self):
        """Should return non-empty list."""
        feats = get_categorical_features()
        assert len(feats) > 0
        assert "job" in feats

    def test_prediction_features_exclude_duration(self):
        """Prediction features should exclude 'duration'."""
        feats = get_prediction_features()
        assert "duration" not in feats
        assert "age" in feats
        assert "job" in feats

    def test_target_name(self):
        """Should return 'subscribe'."""
        assert get_target_name() == "subscribe"

    def test_all_features_non_overlapping(self):
        """Numerical and categorical feature lists should not overlap."""
        num = set(get_numerical_features())
        cat = set(get_categorical_features())
        assert len(num & cat) == 0
