# banksys_sy_wangzhihao

银行营销数据分析与认购预测系统 — 基于 Streamlit 的交互式 Web 应用。

## 功能

1. **数据分析交互页面** — 多维度可视化探索银行营销数据(分布、相关性、认购率分析)
2. **在线预测系统** — 基于离线训练的 ML 模型,通过点选输入客户特征实时预测认购意向

## 技术栈

Python 3.8.5 · Streamlit · pandas · plotly · scikit-learn · Docker · GitHub Actions

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt -r requirements-dev.txt

# 2. 确保 data/train.csv 和 data/test.csv 就位

# 3. 训练模型
python -m src.model_trainer

# 4. 启动应用
streamlit run app.py --server.port 8888
```

打开浏览器访问 `http://localhost:8888`

## Docker

```bash
docker build -t banksys_sy_wangzhihao .
docker run -d --name banksys_sy_wangzhihao -p 8888:8888 banksys_sy_wangzhihao
```

## 项目结构

详见 `standards/00-project-context.md`

## 许可证

MIT
