"""
离线模型训练模块

基于银行营销历史数据训练二分类模型,预测客户是否会认购定期存款。

流程:
1. 加载数据 → 2. 特征工程 → 3. 训练/验证集划分
→ 4. 多模型训练 → 5. 评估对比 → 6. 保存最优模型
"""

import os
import pickle
import warnings
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

from src.data_loader import (
    load_data,
    get_prediction_features,
    get_target_name,
    TRAIN_PATH,
)

warnings.filterwarnings("ignore", category=UserWarning)

# Paths
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.txt")

# Training config
RANDOM_STATE = 42
TEST_SIZE = 0.2
AUC_THRESHOLD = 0.70
ACCURACY_THRESHOLD = 0.75


def build_preprocessor(
    numerical_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    """Build a ColumnTransformer for feature preprocessing.

    Numerical: median imputation + standardization.
    Categorical: most-frequent imputation + one-hot encoding.

    Args:
        numerical_features: List of numerical column names.
        categorical_features: List of categorical column names.

    Returns:
        Fitted or unfitted ColumnTransformer.
    """
    numerical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """Prepare features and target for training.

    Excludes 'id' and 'duration' from feature set (duration is unknown before call).

    Args:
        df: Raw DataFrame from CSV.

    Returns:
        Tuple of (X, y, numerical_feature_names, categorical_feature_names).
    """
    target = get_target_name()
    prediction_features = get_prediction_features()

    # Ensure all prediction features exist
    available_features = [f for f in prediction_features if f in df.columns]

    # Separate numerical and categorical from available features
    from src.data_loader import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

    num_features = [f for f in NUMERICAL_FEATURES if f in available_features]
    cat_features = [f for f in CATEGORICAL_FEATURES if f in available_features]

    X = df[available_features].copy()
    y = df[target].map({"yes": 1, "no": 0})

    if y.isnull().any():
        raise ValueError("Target column contains values other than 'yes'/'no'")

    return X, y, num_features, cat_features


def build_model_pipeline(
    num_features: List[str], cat_features: List[str]
) -> Pipeline:
    """Build the full model pipeline: preprocessing + classifier.

    Args:
        num_features: Numerical feature names.
        cat_features: Categorical feature names.

    Returns:
        Unfitted sklearn Pipeline.
    """
    preprocessor = build_preprocessor(num_features, cat_features)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )
    return pipeline


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    num_features: List[str],
    cat_features: List[str],
) -> Dict[str, Any]:
    """Train multiple models and return evaluation results.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        num_features: Numerical feature names.
        cat_features: Categorical feature names.

    Returns:
        Dict with keys: best_pipeline, metrics, all_results.
    """
    preprocessor = build_preprocessor(num_features, cat_features)

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    best_model_name = None
    best_auc = -1
    best_pipeline = None

    for name, classifier in models.items():
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Metrics
        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
            "auc": round(roc_auc_score(y_test, y_proba), 4),
        }

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

        if metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_model_name = name
            best_pipeline = pipeline

    return {
        "best_pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "best_auc": best_auc,
        "all_results": results,
        "y_test": y_test,
    }


def evaluate_model(model, X_test, y_test) -> Dict[str, float]:
    """Evaluate a trained model on test data.

    Args:
        model: Trained sklearn Pipeline.
        X_test: Test features.
        y_test: True labels (0/1).

    Returns:
        Dict of metric_name → value.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc": round(roc_auc_score(y_test, y_proba), 4),
    }


def save_model(pipeline: Pipeline, feature_names: List[str], path: str = None):
    """Save trained pipeline and metadata to disk.

    Args:
        pipeline: Trained sklearn Pipeline.
        feature_names: Ordered list of feature names used in training.
        path: Output path (defaults to models/best_model.pkl).
    """
    if path is None:
        path = MODEL_PATH

    os.makedirs(os.path.dirname(path), exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "sklearn_version": __import__("sklearn").__version__,
    }

    with open(path, "wb") as f:
        pickle.dump(artifact, f)

    return path


def check_model_quality(metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Check if model meets quality thresholds.

    Args:
        metrics: Dict with at least 'auc' and 'accuracy' keys.

    Returns:
        Tuple of (passed: bool, failures: list of str).
    """
    failures = []
    if metrics.get("auc", 0) < AUC_THRESHOLD:
        failures.append(
            f"AUC {metrics['auc']:.4f} < {AUC_THRESHOLD} threshold"
        )
    if metrics.get("accuracy", 0) < ACCURACY_THRESHOLD:
        failures.append(
            f"Accuracy {metrics['accuracy']:.4f} < {ACCURACY_THRESHOLD} threshold"
        )
    return len(failures) == 0, failures


def run_training(data_path: str = None, output_path: str = None) -> Dict[str, Any]:
    """Run the full training pipeline end-to-end.

    Args:
        data_path: Path to training CSV (default: data/train.csv).
        output_path: Path to save model (default: models/best_model.pkl).

    Returns:
        Dict with training results including best_model_name, metrics, and save path.
    """
    # 1. Load data
    df = load_data(data_path)
    print(f"[1/5] 数据加载完成: {len(df)} 行, {len(df.columns)} 列")

    # 2. Prepare features
    X, y, num_features, cat_features = prepare_data(df)
    print(
        f"[2/5] 特征准备完成: {len(num_features)} 数值特征 + "
        f"{len(cat_features)} 分类特征"
    )

    # 3. Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"[3/5] 数据划分: 训练 {len(X_train)} / 测试 {len(X_test)} "
        f"(正样本率 {y_train.mean():.2%})"
    )

    # 4. Train and evaluate
    results = train_and_evaluate(
        X_train, y_train, X_test, y_test, num_features, cat_features
    )
    print(f"[4/5] 模型训练完成")

    # Print per-model results
    for name, data in results["all_results"].items():
        m = data["metrics"]
        print(
            f"  {name:25s} AUC={m['auc']:.4f}  Acc={m['accuracy']:.4f}  "
            f"Precision={m['precision']:.4f}  Recall={m['recall']:.4f}  F1={m['f1']:.4f}"
        )

    # 5. Check quality and save
    best_metrics = results["all_results"][results["best_model_name"]]["metrics"]
    passed, failures = check_model_quality(best_metrics)

    if passed:
        save_path = save_model(
            results["best_pipeline"], list(X.columns), output_path
        )
        print(f"[5/5] ✅ 模型已保存: {save_path}")
    else:
        save_path = save_model(
            results["best_pipeline"], list(X.columns), output_path
        )
        print(f"[5/5] ⚠️ 模型已保存但不符合质量门禁: {save_path}")
        for f in failures:
            print(f"  - {f}")

    return {
        "best_model_name": results["best_model_name"],
        "best_auc": results["best_auc"],
        "all_results": results["all_results"],
        "save_path": save_path if passed or output_path else None,
        "quality_passed": passed,
        "feature_names": list(X.columns),
        "num_features": num_features,
        "cat_features": cat_features,
    }


# ============================================================
# CLI entry point
# ============================================================
if __name__ == "__main__":
    run_training()
