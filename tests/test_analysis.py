"""
数据分析模块 — 单元测试

测试数据概览、分布图数据、关系图数据、认购率分析、相关性计算的逻辑层。
注:Streamlit 渲染函数需要 streamlit context,这里主要测试数据变换逻辑。
"""

import pandas as pd
import numpy as np
import pytest
import plotly.graph_objects as go

from src.data_loader import (
    get_data_summary,
    get_missing_report,
    get_feature_stats,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
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


class TestDataOverview:
    """Tests for data overview functions."""

    def test_summary_correct_row_count(self, sample_df):
        """Summary should report correct row count."""
        summary = get_data_summary(sample_df)
        assert summary["rows"] == 500

    def test_summary_correct_column_count(self, sample_df):
        """Summary should report correct column count."""
        summary = get_data_summary(sample_df)
        assert summary["columns"] == 22  # id + 20 features + target

    def test_summary_has_positive_rate(self, sample_df):
        """Summary should include positive rate."""
        summary = get_data_summary(sample_df)
        assert "positive_rate" in summary
        assert 0 < summary["positive_rate"] < 100

    def test_summary_target_distribution_sums_to_total(self, sample_df):
        """Target distribution should sum to total rows."""
        summary = get_data_summary(sample_df)
        total = sum(summary["target_distribution"].values())
        assert total == 500

    def test_missing_report_all_columns(self, sample_df):
        """Missing report should cover all columns."""
        df = sample_df.copy()
        df.loc[0:5, "age"] = None
        report = get_missing_report(df)
        assert len(report) == len(df.columns)

    def test_missing_report_counts_missing(self, sample_df):
        """Missing report should count missing values correctly."""
        df = sample_df.copy()
        df.loc[0:9, "job"] = None
        report = get_missing_report(df)
        job_row = report[report["feature"] == "job"]
        assert job_row["missing_count"].values[0] == 10


class TestFeatureStats:
    """Tests for feature statistics."""

    def test_numerical_stats_not_empty(self, sample_df):
        """Numerical stats should be non-empty."""
        num_stats, _ = get_feature_stats(sample_df)
        assert not num_stats.empty

    def test_categorical_stats_not_empty(self, sample_df):
        """Categorical stats should be non-empty."""
        _, cat_stats = get_feature_stats(sample_df)
        assert not cat_stats.empty

    def test_numerical_stats_has_mean(self, sample_df):
        """Numerical stats should include mean."""
        num_stats, _ = get_feature_stats(sample_df)
        assert "mean" in num_stats.index

    def test_feature_stats_excludes_id(self, sample_df):
        """Stats should not include the id column."""
        num_stats, cat_stats = get_feature_stats(sample_df)
        stat_cols = set(num_stats.columns) | set(cat_stats.columns)
        assert "id" not in stat_cols


class TestCorrelationComputation:
    """Tests for correlation computation logic."""

    def test_correlation_matrix_shape(self, sample_df):
        """Correlation matrix should be square with correct size."""
        num_cols = [c for c in NUMERICAL_FEATURES if c in sample_df.columns]
        corr = sample_df[num_cols].corr()
        assert corr.shape[0] == corr.shape[1]
        assert corr.shape[0] == len(num_cols)

    def test_correlation_with_target(self, sample_df):
        """Should be able to correlate with encoded target."""
        df = sample_df.copy()
        df["subscribe_num"] = (df["subscribe"] == "yes").astype(int)
        num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
        all_cols = ["subscribe_num"] + num_cols
        corr = df[all_cols].corr()
        assert "subscribe_num" in corr.index
        assert all(-1 <= v <= 1 for v in corr["subscribe_num"].drop("subscribe_num"))

    def test_correlation_values_in_range(self, sample_df):
        """All correlation values should be between -1 and 1."""
        num_cols = [c for c in NUMERICAL_FEATURES if c in sample_df.columns]
        corr = sample_df[num_cols].corr()
        for col in corr.columns:
            for idx in corr.index:
                assert -1 <= corr.loc[idx, col] <= 1


class TestGroupedAnalysis:
    """Tests for grouped subscription analysis logic."""

    def test_subscription_rate_by_job(self, sample_df):
        """Should compute subscription rate per job category."""
        grouped = sample_df.groupby("job").agg(
            total=("subscribe", "count"),
            yes_count=("subscribe", lambda x: (x == "yes").sum()),
        )
        grouped["rate"] = (grouped["yes_count"] / grouped["total"] * 100).round(1)

        total_jobs = grouped["total"].sum()
        total_yes = grouped["yes_count"].sum()
        assert total_jobs == 500
        assert total_yes <= 500

    def test_subscription_rate_values_in_range(self, sample_df):
        """All subscription rates should be between 0 and 100."""
        grouped = sample_df.groupby("job").agg(
            total=("subscribe", "count"),
            yes_count=("subscribe", lambda x: (x == "yes").sum()),
        )
        grouped["rate"] = (grouped["yes_count"] / grouped["total"] * 100).round(1)
        assert all(0 <= r <= 100 for r in grouped["rate"])

    def test_age_binning(self, sample_df):
        """Age binning should produce correct groups."""
        df = sample_df.copy()
        df["age_group"] = pd.cut(
            df["age"],
            bins=[17, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        assert df["age_group"].notna().all()
        # Each row should be assigned to exactly one group
        assert len(df["age_group"].unique()) <= 6
