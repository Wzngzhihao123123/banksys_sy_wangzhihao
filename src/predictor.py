"""
在线预测模块

加载离线训练的模型,通过 Streamlit 点选表单接收用户输入,
实时预测客户是否会认购定期存款,并展示预测概率与对比分析。
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline

from src.data_loader import (
    get_prediction_features,
    load_data,
)

# Model path
MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")

# ---- Feature value options for input widgets ----

JOB_OPTIONS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown",
]

MARITAL_OPTIONS = ["divorced", "married", "single", "unknown"]

EDUCATION_OPTIONS = [
    "basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate",
    "professional.course", "university.degree", "unknown",
]

MONTH_OPTIONS = [
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "oct", "nov", "dec",
]

DAY_OF_WEEK_OPTIONS = ["mon", "tue", "wed", "thu", "fri"]

POUTCOME_OPTIONS = ["failure", "nonexistent", "success"]

YES_NO_UNKNOWN = ["no", "yes", "unknown"]

# Feature value ranges for defaults
FEATURE_DEFAULTS = {
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

# Feature help/tooltip text
FEATURE_HELP = {
    "age": "客户年龄",
    "job": "职业类型",
    "marital": "婚姻状况",
    "education": "教育水平",
    "default": "是否有信用违约记录",
    "housing": "是否有住房贷款",
    "loan": "是否有个人贷款",
    "contact": "接触方式 (手机/座机)",
    "month": "最近一次接触的月份",
    "day_of_week": "最近一次接触的星期",
    "campaign": "本次营销活动中对该客户的接触次数",
    "pdays": "距上次营销活动过去的天数 (999=从未接触过)",
    "previous": "本次营销活动前接触次数",
    "poutcome": "上次营销活动的结果",
    "emp_var_rate": "就业变化率 (季度指标)",
    "cons_price_index": "消费者物价指数 (月度指标)",
    "cons_conf_index": "消费者信心指数 (月度指标)",
    "lending_rate3m": "3个月 Euribor 利率 (日度指标)",
    "nr_employed": "就业人数 (季度指标,千人)",
}


@st.cache_resource
def load_model() -> Tuple[Any, List[str], Any]:
    """Load the trained model from disk. Cached to avoid repeated I/O.

    Returns:
        Tuple of (model_pipeline, feature_names, preprocessor).

    Raises:
        FileNotFoundError: If model file does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"模型文件未找到: {MODEL_PATH}\n"
            f"请先运行训练: python -m src.model_trainer"
        )

    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    pipeline = artifact["pipeline"]
    feature_names = artifact["feature_names"]

    return pipeline, feature_names, None


def predict(
    model: Pipeline,
    preprocessor: Any,
    user_input: Dict[str, Any],
    feature_names: List[str],
) -> Tuple[str, float]:
    """Run prediction on a single user input.

    Args:
        model: Trained sklearn Pipeline.
        preprocessor: Unused (pipeline includes preprocessing).
        user_input: Dict of feature_name → value.
        feature_names: Ordered feature list matching training.

    Returns:
        Tuple of (prediction_label: '会认购' | '不会认购', probability: float).
    """
    # Build a single-row DataFrame
    row = {feat: user_input.get(feat, 0) for feat in feature_names}
    df = pd.DataFrame([row])

    # Predict
    proba = model.predict_proba(df)[0]
    pred_class = model.predict(df)[0]

    # proba[0] = probability of class 0 ("no"), proba[1] = probability of class 1 ("yes")
    yes_prob = proba[1] if len(proba) > 1 else proba[0]

    if pred_class == 1:
        label = "会认购 ✅"
    else:
        label = "不会认购 ❌"

    return label, float(yes_prob)


def get_feature_inputs() -> Optional[Dict[str, Any]]:
    """Render the feature input form using Streamlit widgets.

    Returns:
        Dict of feature_name → value, or None if validation fails.
    """
    user_input = {}

    # --- Demographic group ---
    st.markdown("#### 👤 人口统计信息")
    col1, col2, col3 = st.columns(3)

    with col1:
        user_input["age"] = st.slider(
            "年龄",
            min_value=18,
            max_value=95,
            value=FEATURE_DEFAULTS["age"],
            help=FEATURE_HELP["age"],
        )

    with col2:
        user_input["job"] = st.selectbox(
            "职业",
            options=JOB_OPTIONS,
            index=JOB_OPTIONS.index(FEATURE_DEFAULTS["job"]),
            help=FEATURE_HELP["job"],
        )

    with col3:
        user_input["marital"] = st.selectbox(
            "婚姻状况",
            options=MARITAL_OPTIONS,
            index=MARITAL_OPTIONS.index(FEATURE_DEFAULTS["marital"]),
            help=FEATURE_HELP["marital"],
        )

    user_input["education"] = st.selectbox(
        "教育水平",
        options=EDUCATION_OPTIONS,
        index=EDUCATION_OPTIONS.index(FEATURE_DEFAULTS["education"]),
        help=FEATURE_HELP["education"],
    )

    st.markdown("---")

    # --- Financial group ---
    st.markdown("#### 💰 财务状况")
    col1, col2, col3 = st.columns(3)

    with col1:
        user_input["default"] = st.radio(
            "信用违约",
            options=YES_NO_UNKNOWN,
            index=YES_NO_UNKNOWN.index(FEATURE_DEFAULTS["default"]),
            horizontal=True,
            help=FEATURE_HELP["default"],
        )

    with col2:
        user_input["housing"] = st.radio(
            "住房贷款",
            options=YES_NO_UNKNOWN,
            index=YES_NO_UNKNOWN.index(FEATURE_DEFAULTS["housing"]),
            horizontal=True,
            help=FEATURE_HELP["housing"],
        )

    with col3:
        user_input["loan"] = st.radio(
            "个人贷款",
            options=YES_NO_UNKNOWN,
            index=YES_NO_UNKNOWN.index(FEATURE_DEFAULTS["loan"]),
            horizontal=True,
            help=FEATURE_HELP["loan"],
        )

    st.markdown("---")

    # --- Campaign contact group ---
    st.markdown("#### 📞 营销接触信息")
    col1, col2, col3 = st.columns(3)

    with col1:
        user_input["contact"] = st.radio(
            "接触方式",
            options=["cellular", "telephone"],
            index=0,
            horizontal=True,
            help=FEATURE_HELP["contact"],
        )

    with col2:
        user_input["month"] = st.selectbox(
            "接触月份",
            options=MONTH_OPTIONS,
            index=MONTH_OPTIONS.index(FEATURE_DEFAULTS["month"]),
            help=FEATURE_HELP["month"],
        )

    with col3:
        user_input["day_of_week"] = st.selectbox(
            "接触星期",
            options=DAY_OF_WEEK_OPTIONS,
            index=DAY_OF_WEEK_OPTIONS.index(FEATURE_DEFAULTS["day_of_week"]),
            help=FEATURE_HELP["day_of_week"],
        )

    col1, col2, col3 = st.columns(3)

    with col1:
        user_input["campaign"] = st.number_input(
            "本次接触次数",
            min_value=1,
            max_value=50,
            value=FEATURE_DEFAULTS["campaign"],
            help=FEATURE_HELP["campaign"],
        )

    with col2:
        user_input["pdays"] = st.number_input(
            "距上次活动天数",
            min_value=0,
            max_value=999,
            value=FEATURE_DEFAULTS["pdays"],
            help=FEATURE_HELP["pdays"],
        )

    with col3:
        user_input["previous"] = st.number_input(
            "历史接触次数",
            min_value=0,
            max_value=10,
            value=FEATURE_DEFAULTS["previous"],
            help=FEATURE_HELP["previous"],
        )

    user_input["poutcome"] = st.selectbox(
        "上次活动结果",
        options=POUTCOME_OPTIONS,
        index=POUTCOME_OPTIONS.index(FEATURE_DEFAULTS["poutcome"]),
        help=FEATURE_HELP["poutcome"],
    )

    st.markdown("---")

    # --- Economic indicators group ---
    st.markdown("#### 📊 社会经济指标")
    col1, col2, col3 = st.columns(3)

    with col1:
        user_input["emp_var_rate"] = st.number_input(
            "就业变化率",
            min_value=-5.0,
            max_value=5.0,
            value=FEATURE_DEFAULTS["emp_var_rate"],
            step=0.1,
            format="%.1f",
            help=FEATURE_HELP["emp_var_rate"],
        )

    with col2:
        user_input["cons_price_index"] = st.number_input(
            "消费者物价指数",
            min_value=85.0,
            max_value=100.0,
            value=FEATURE_DEFAULTS["cons_price_index"],
            step=0.1,
            format="%.2f",
            help=FEATURE_HELP["cons_price_index"],
        )

    with col3:
        user_input["cons_conf_index"] = st.number_input(
            "消费者信心指数",
            min_value=-60.0,
            max_value=-20.0,
            value=FEATURE_DEFAULTS["cons_conf_index"],
            step=0.1,
            format="%.1f",
            help=FEATURE_HELP["cons_conf_index"],
        )

    col1, col2 = st.columns(2)

    with col1:
        user_input["lending_rate3m"] = st.number_input(
            "3个月利率 (Euribor)",
            min_value=0.0,
            max_value=10.0,
            value=FEATURE_DEFAULTS["lending_rate3m"],
            step=0.01,
            format="%.2f",
            help=FEATURE_HELP["lending_rate3m"],
        )

    with col2:
        user_input["nr_employed"] = st.number_input(
            "就业人数 (千人)",
            min_value=4500.0,
            max_value=5500.0,
            value=FEATURE_DEFAULTS["nr_employed"],
            step=1.0,
            format="%.1f",
            help=FEATURE_HELP["nr_employed"],
        )

    return user_input


def display_prediction_result(
    label: str, probability: float, user_input: Dict[str, Any]
):
    """Display the prediction result with supporting info.

    Args:
        label: Prediction label (e.g., '会认购 ✅').
        probability: Probability of positive class.
        user_input: The feature values that were submitted.
    """
    st.markdown("---")
    st.markdown("### 📊 预测结果")

    # Main result
    col1, col2 = st.columns([1, 2])

    with col1:
        if "会认购" in label:
            st.success(f"## {label}")
        else:
            st.warning(f"## {label}")

        st.metric("认购概率", f"{probability:.1%}")

        # Gauge-like probability bar
        st.progress(min(float(probability), 1.0))

    with col2:
        st.markdown("**概率解读**")
        if probability >= 0.7:
            st.info("🔴 高概率认购 — 建议优先跟进该客户")
        elif probability >= 0.4:
            st.info("🟡 中等概率 — 可考虑再次接触营销")
        else:
            st.info("🟢 低概率认购 — 可暂缓营销资源投入")

    st.markdown("---")

    # Try to show comparison with historical data
    try:
        df = load_data()
        df_yes = df[df["subscribe"] == "yes"]

        st.markdown("### 📈 与历史认购者对比")

        comparisons = []

        # Age comparison
        if "age" in user_input and "age" in df_yes.columns:
            user_age = user_input["age"]
            pct = (df_yes["age"] <= user_age).mean() * 100
            comparisons.append(
                {
                    "特征": "年龄",
                    "您的值": str(user_age),
                    "历史认购者均值": f"{df_yes['age'].mean():.0f}",
                    "所处百分位": f"Top {100 - pct:.0f}%" if pct > 50 else f"Bottom {pct:.0f}%",
                }
            )

        # Campaign comparison
        if "campaign" in user_input and "campaign" in df_yes.columns:
            user_campaign = user_input["campaign"]
            avg_campaign = df_yes["campaign"].mean()
            comparisons.append(
                {
                    "特征": "接触次数",
                    "您的值": str(user_campaign),
                    "历史认购者均值": f"{avg_campaign:.1f}",
                    "对比": "低于均值" if user_campaign <= avg_campaign else "高于均值",
                }
            )

        # Previous comparison
        if "previous" in user_input and "previous" in df_yes.columns:
            user_prev = user_input["previous"]
            pct_prev = (df_yes["previous"] == 0).mean() * 100
            comparisons.append(
                {
                    "特征": "历史接触",
                    "您的值": str(user_prev),
                    "认购者中从未接触比例": f"{pct_prev:.1f}%",
                    "对比": "与多数认购者一致" if user_prev == 0 else "高于多数认购者",
                }
            )

        if comparisons:
            st.table(pd.DataFrame(comparisons))

    except FileNotFoundError:
        st.caption("💡 加载历史数据后可与历史认购者对比分析")
    except Exception:
        st.caption("💡 历史对比数据暂不可用")
