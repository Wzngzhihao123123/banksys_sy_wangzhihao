"""
银行营销数据分析与认购预测系统 - Streamlit 应用入口

两个核心功能:
1. 数据分析交互页面 — 多维度可视化探索
2. 在线预测系统 — 基于离线训练模型,点选输入特征实时预测
"""

import streamlit as st

st.set_page_config(
    page_title="银行营销分析预测系统",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    st.sidebar.title("🏦 银行营销系统")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "导航",
        ["📊 数据分析", "🔮 在线预测"],
        help="选择功能模块",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("banksys_sy_wangzhihao")
    st.sidebar.caption("Port: 8888")

    if page == "📊 数据分析":
        show_analysis_page()
    elif page == "🔮 在线预测":
        show_prediction_page()


def show_analysis_page():
    """数据分析交互页面"""
    st.title("📊 银行营销数据分析")

    try:
        from src.data_loader import load_data, get_data_summary
        from src.analysis import (
            render_data_overview,
            render_distribution,
            render_relationship,
            render_subscription_analysis,
            render_correlation_heatmap,
        )

        df = load_data()

        # Sidebar filters
        st.sidebar.header("🔍 数据筛选")

        # Job filter
        all_jobs = sorted(df["job"].dropna().unique())
        selected_jobs = st.sidebar.multiselect(
            "职业", all_jobs, default=[], help="按职业筛选"
        )

        # Marital filter
        all_marital = sorted(df["marital"].dropna().unique())
        selected_marital = st.sidebar.multiselect(
            "婚姻状况", all_marital, default=[], help="按婚姻状况筛选"
        )

        # Education filter
        all_education = sorted(df["education"].dropna().unique())
        selected_education = st.sidebar.multiselect(
            "教育水平", all_education, default=[], help="按教育水平筛选"
        )

        # Age range filter
        age_min, age_max = int(df["age"].min()), int(df["age"].max())
        selected_age = st.sidebar.slider(
            "年龄范围",
            age_min,
            age_max,
            (age_min, age_max),
            help="按年龄范围筛选",
        )

        # Apply filters
        filtered_df = df.copy()
        if selected_jobs:
            filtered_df = filtered_df[filtered_df["job"].isin(selected_jobs)]
        if selected_marital:
            filtered_df = filtered_df[filtered_df["marital"].isin(selected_marital)]
        if selected_education:
            filtered_df = filtered_df[filtered_df["education"].isin(selected_education)]
        filtered_df = filtered_df[
            (filtered_df["age"] >= selected_age[0])
            & (filtered_df["age"] <= selected_age[1])
        ]

        st.sidebar.markdown("---")
        st.sidebar.metric("筛选后数据量", f"{len(filtered_df):,} 行")

        # Render sections
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["📋 数据概览", "📈 特征分布", "🔗 特征关系", "🎯 认购分析", "🔥 相关性"]
        )

        with tab1:
            render_data_overview(filtered_df)

        with tab2:
            render_distribution(filtered_df)

        with tab3:
            render_relationship(filtered_df)

        with tab4:
            render_subscription_analysis(filtered_df)

        with tab5:
            render_correlation_heatmap(filtered_df)

    except FileNotFoundError as e:
        st.error(f"❌ 数据文件未找到: {e}")
        st.info(
            "请确保 `data/train.csv` 和 `data/test.csv` 已放置在项目根目录下的 `data/` 文件夹中。"
        )
    except Exception as e:
        st.error(f"❌ 加载失败: {e}")


def show_prediction_page():
    """在线预测页面"""
    st.title("🔮 在线认购预测")

    try:
        from src.predictor import (
            load_model,
            predict,
            get_feature_inputs,
            display_prediction_result,
        )

        model, feature_names, preprocessor = load_model()

        st.markdown("### 请输入客户特征")
        st.markdown("*请在下表中填写各项特征,完成后点击「开始预测」按钮*")

        user_input = get_feature_inputs()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            predict_btn = st.button(
                "🔮 开始预测",
                type="primary",
                use_container_width=True,
                disabled=user_input is None,
            )

        if predict_btn and user_input is not None:
            result, probability = predict(model, preprocessor, user_input, feature_names)
            display_prediction_result(result, probability, user_input)

    except FileNotFoundError:
        st.warning("⚠️ 模型文件未找到,请先运行训练脚本")
        st.code("python -m src.model_trainer", language="bash")
        st.info(
            "运行上述命令后,模型将保存至 `models/best_model.pkl`,然后刷新本页面即可使用预测功能。"
        )
    except Exception as e:
        st.error(f"❌ 预测系统错误: {e}")


if __name__ == "__main__":
    main()
