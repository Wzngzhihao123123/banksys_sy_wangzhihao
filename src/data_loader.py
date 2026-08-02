"""
数据加载与预处理模块

负责加载银行营销 CSV 数据,提供基础统计信息与数据类型识别。
"""

import os
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Default data path relative to project root
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

# Feature definitions
TARGET = "subscribe"

NUMERICAL_FEATURES = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# Features available for prediction (excluding duration — unknown before call)
PREDICTION_FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]

ID_COLUMN = "id"


def load_data(path: str = None) -> pd.DataFrame:
    """Load training data from CSV.

    Args:
        path: Optional path to CSV. Defaults to data/train.csv relative to project root.

    Returns:
        DataFrame with the training data.

    Raises:
        FileNotFoundError: If the data file does not exist.
    """
    if path is None:
        path = TRAIN_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"数据文件未找到: {path}\n"
            f"请将 train.csv 放置到 {DATA_DIR} 目录下。"
        )

    df = pd.read_csv(path)
    return df


def load_test_data(path: str = None) -> pd.DataFrame:
    """Load test data from CSV."""
    if path is None:
        path = TEST_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"测试数据文件未找到: {path}")

    return pd.read_csv(path)


def get_data_summary(df: pd.DataFrame) -> Dict:
    """Generate a summary dictionary for the dataset.

    Args:
        df: Input DataFrame.

    Returns:
        Dict with keys: rows, columns, missing_total, missing_pct,
        numerical_count, categorical_count, target_distribution.
    """
    summary = {
        "rows": len(df),
        "columns": len(df.columns),
        "missing_total": int(df.isnull().sum().sum()),
        "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
        "numerical_count": len([c for c in NUMERICAL_FEATURES if c in df.columns]),
        "categorical_count": len([c for c in CATEGORICAL_FEATURES if c in df.columns]),
    }

    if TARGET in df.columns:
        target_counts = df[TARGET].value_counts().to_dict()
        summary["target_distribution"] = target_counts
        summary["positive_rate"] = round(
            target_counts.get("yes", 0) / len(df) * 100, 2
        )

    return summary


def get_numerical_features() -> List[str]:
    """Return list of numerical feature names."""
    return NUMERICAL_FEATURES.copy()


def get_categorical_features() -> List[str]:
    """Return list of categorical feature names."""
    return CATEGORICAL_FEATURES.copy()


def get_prediction_features() -> List[str]:
    """Return features used for prediction (excludes duration)."""
    return PREDICTION_FEATURES.copy()


def get_target_name() -> str:
    """Return the target column name."""
    return TARGET


def get_missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Generate per-column missing value report.

    Args:
        df: Input DataFrame.

    Returns:
        DataFrame with columns: feature, missing_count, missing_pct, dtype.
    """
    report = []
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        report.append(
            {
                "feature": col,
                "missing_count": missing,
                "missing_pct": round(missing / len(df) * 100, 2),
                "dtype": str(df[col].dtype),
            }
        )
    return pd.DataFrame(report)


def get_feature_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate descriptive statistics for numerical and categorical features.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (numerical_stats_df, categorical_stats_df).
    """
    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]

    num_stats = df[num_cols].describe() if num_cols else pd.DataFrame()
    cat_stats = df[cat_cols].describe() if cat_cols else pd.DataFrame()

    return num_stats, cat_stats
