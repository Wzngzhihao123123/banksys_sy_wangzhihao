"""
数据分析与可视化模块

为 Streamlit 页面提供多维度数据探索图表:
- 数据概览(行数/列数/缺失值/数据类型)
- 特征分布(直方图/箱线图/柱状图)
- 特征关系(散点图/分组箱线图/堆叠柱状图)
- 认购率分析(整体 + 按特征分组)
- 相关性热力图
"""

from typing import List

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data_loader import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET,
    get_data_summary,
    get_missing_report,
    get_feature_stats,
)


def render_data_overview(df: pd.DataFrame):
    """Render data overview tab: shape, dtypes, missing values, basic stats.

    Args:
        df: Filtered DataFrame to display.
    """
    summary = get_data_summary(df)

    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("数据行数", f"{summary['rows']:,}")
    with col2:
        st.metric("特征列数", summary["columns"])
    with col3:
        st.metric("缺失值总数", f"{summary['missing_total']:,}")
    with col4:
        st.metric("数值特征", summary["numerical_count"])
    with col5:
        st.metric("分类特征", summary["categorical_count"])

    if "positive_rate" in summary:
        st.metric("认购率 (yes)", f"{summary['positive_rate']}%")

    st.markdown("---")

    # Missing value report
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.subheader("缺失值报告")
        missing_report = get_missing_report(df)
        missing_report = missing_report[missing_report["missing_count"] > 0]
        if missing_report.empty:
            st.success("✅ 数据集无缺失值")
        else:
            st.dataframe(missing_report, use_container_width=True, hide_index=True)

    with col_right:
        st.subheader("数据类型分布")
        dtype_counts = df.dtypes.astype(str).value_counts().reset_index()
        dtype_counts.columns = ["数据类型", "数量"]
        fig = px.pie(
            dtype_counts, values="数量", names="数据类型", hole=0.4
        )
        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Data sample
    st.markdown("---")
    st.subheader("数据预览 (前 100 行)")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)


def render_distribution(df: pd.DataFrame):
    """Render feature distribution tab.

    Args:
        df: Filtered DataFrame to display.
    """
    st.subheader("特征分布探查")

    # Feature selection
    all_features = [c for c in df.columns if c not in (TARGET, "id")]
    default_idx = all_features.index("age") if "age" in all_features else 0

    feature = st.selectbox(
        "选择特征",
        all_features,
        index=default_idx,
        help="选择一个特征查看其分布情况",
    )

    if feature is None:
        return

    col1, col2 = st.columns([3, 1])

    with col2:
        st.markdown("**特征信息**")
        st.metric("唯一值数", df[feature].nunique())
        st.metric("缺失数", int(df[feature].isnull().sum()))
        if feature in NUMERICAL_FEATURES:
            st.metric("均值", f"{df[feature].mean():.2f}")
            st.metric("标准差", f"{df[feature].std():.2f}")

    with col1:
        if feature in NUMERICAL_FEATURES:
            # Numerical: histogram + boxplot subplot
            fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=("直方图", "箱线图"),
                column_widths=[0.6, 0.4],
            )

            fig.add_trace(
                go.Histogram(
                    x=df[feature].dropna(),
                    nbinsx=50,
                    marker_color="#1f77b4",
                    name="频数",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Box(
                    x=df[feature].dropna(),
                    marker_color="#1f77b4",
                    name="箱线图",
                ),
                row=1,
                col=2,
            )

            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Min/Max/Percentiles
            st.caption(
                f"最小值: {df[feature].min():.2f} | "
                f"P25: {df[feature].quantile(0.25):.2f} | "
                f"P50: {df[feature].quantile(0.50):.2f} | "
                f"P75: {df[feature].quantile(0.75):.2f} | "
                f"最大值: {df[feature].max():.2f}"
            )
        else:
            # Categorical: bar chart
            value_counts = df[feature].value_counts().nlargest(20)
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={"x": feature, "y": "数量"},
                color=value_counts.values,
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig, use_container_width=True)


def render_relationship(df: pd.DataFrame):
    """Render feature relationship tab.

    Args:
        df: Filtered DataFrame to display.
    """
    st.subheader("特征关系探查")

    all_features = [c for c in df.columns if c not in (TARGET, "id")]

    col_x, col_y = st.columns(2)
    with col_x:
        x_feature = st.selectbox(
            "X 轴特征",
            all_features,
            index=all_features.index("age") if "age" in all_features else 0,
        )
    with col_y:
        y_feature = st.selectbox(
            "Y 轴特征",
            all_features,
            index=all_features.index("duration") if "duration" in all_features else 0,
        )

    if x_feature is None or y_feature is None:
        return

    x_is_num = x_feature in NUMERICAL_FEATURES
    y_is_num = y_feature in NUMERICAL_FEATURES

    # Color by target
    color_col = TARGET if TARGET in df.columns else None

    if x_is_num and y_is_num:
        # Scatter plot
        fig = px.scatter(
            df.sample(min(2000, len(df))),
            x=x_feature,
            y=y_feature,
            color=color_col,
            opacity=0.6,
            labels={x_feature: x_feature, y_feature: y_feature},
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    elif not x_is_num and not y_is_num:
        # Stacked bar: count of y grouped by x
        cross_tab = pd.crosstab(df[x_feature], df[y_feature])
        # Limit to top categories
        top_x = df[x_feature].value_counts().nlargest(10).index
        cross_tab = cross_tab.loc[cross_tab.index.isin(top_x)]
        fig = px.bar(
            cross_tab,
            barmode="stack",
            labels={"x": x_feature, "value": "数量", "variable": y_feature},
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    else:
        # One numerical, one categorical: grouped box
        cat_feat = x_feature if not x_is_num else y_feature
        num_feat = y_feature if not x_is_num else x_feature

        # Limit categories for readability
        top_cats = df[cat_feat].value_counts().nlargest(10).index
        plot_df = df[df[cat_feat].isin(top_cats)]

        fig = px.box(
            plot_df,
            x=cat_feat,
            y=num_feat,
            color=color_col,
            labels={cat_feat: cat_feat, num_feat: num_feat},
        )
        fig.update_layout(height=500, margin=dict(l=0, r=0, t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)


def render_subscription_analysis(df: pd.DataFrame):
    """Render subscription rate analysis tab.

    Args:
        df: Filtered DataFrame to display.
    """
    st.subheader("认购率分析")

    if TARGET not in df.columns:
        st.warning("数据中无 subscribe 列,无法展示认购分析")
        return

    # Overall rate
    overall_rate = (df[TARGET] == "yes").mean() * 100
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("整体认购率", f"{overall_rate:.1f}%")
    with col2:
        st.metric("认购人数", f"{(df[TARGET] == 'yes').sum():,}")
    with col3:
        st.metric("未认购人数", f"{(df[TARGET] == 'no').sum():,}")

    st.markdown("---")

    # Subscription rate by key features
    st.markdown("### 按特征分组认购率")

    cat_features = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    selected_feat = st.selectbox(
        "选择分组特征",
        cat_features,
        index=cat_features.index("job") if "job" in cat_features else 0,
    )

    if selected_feat:
        # Calculate rates
        grouped = df.groupby(selected_feat).agg(
            total=("subscribe", "count"),
            yes_count=("subscribe", lambda x: (x == "yes").sum()),
        )
        grouped["认购率"] = (grouped["yes_count"] / grouped["total"] * 100).round(1)
        grouped = grouped.sort_values("认购率", ascending=False)

        # Bar chart
        fig = px.bar(
            grouped.reset_index(),
            x=selected_feat,
            y="认购率",
            color="认购率",
            color_continuous_scale="RdYlGn",
            text="认购率",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(
            y=overall_rate,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"整体 ({overall_rate:.1f}%)",
        )
        fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=60))
        st.plotly_chart(fig, use_container_width=True)

        # Show table
        st.dataframe(
            grouped.reset_index(),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("---")

    # Age groups vs subscription
    st.markdown("### 年龄与认购率")
    if "age" in df.columns:
        df_temp = df.copy()
        df_temp["年龄段"] = pd.cut(
            df_temp["age"],
            bins=[17, 25, 35, 45, 55, 65, 100],
            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        )
        age_grouped = df_temp.groupby("年龄段").agg(
            total=("subscribe", "count"),
            yes_count=("subscribe", lambda x: (x == "yes").sum()),
        )
        age_grouped["认购率"] = (
            age_grouped["yes_count"] / age_grouped["total"] * 100
        ).round(1)

        fig = px.line(
            age_grouped.reset_index(),
            x="年龄段",
            y="认购率",
            markers=True,
            line_shape="spline",
        )
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        fig.update_traces(line=dict(color="#d62728", width=2))
        st.plotly_chart(fig, use_container_width=True)


def render_correlation_heatmap(df: pd.DataFrame):
    """Render correlation heatmap for numerical features.

    Args:
        df: Filtered DataFrame to display.
    """
    st.subheader("数值特征相关性")

    num_cols = [c for c in NUMERICAL_FEATURES if c in df.columns]
    if len(num_cols) < 2:
        st.warning("数值特征不足,无法计算相关性")
        return

    # Include target as numeric
    corr_cols = num_cols.copy()
    if TARGET in df.columns:
        df_temp = df.copy()
        df_temp[TARGET] = (df_temp[TARGET] == "yes").astype(int)
        corr_cols = [TARGET] + num_cols

    corr_matrix = df_temp[corr_cols].corr()

    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        height=550,
        margin=dict(l=0, r=0, t=20, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Top correlations with target
    if TARGET in df.columns:
        st.markdown("### 与认购的相关性 Top-5")
        target_corr = corr_matrix[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
        top5 = target_corr.head(5)

        fig = px.bar(
            x=top5.index,
            y=top5.values,
            labels={"x": "特征", "y": "相关系数"},
            color=top5.values,
            color_continuous_scale="RdBu_r",
        )
        fig.update_layout(height=300, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
