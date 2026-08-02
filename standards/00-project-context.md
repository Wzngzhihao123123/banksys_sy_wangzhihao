# 00 · 项目上下文 〔本项目活记忆 · AI 维护〕

> **作用**:这是项目的"身份档案"。AI 接管项目时先读这里,了解项目目标、技术栈、目录、部署取值。
> **更新时机**:架构、技术栈、目录结构、端口、部署目录、重要约束变化时更新。
> **填写方式**:把 `<...>` 替换成真实内容;用不到的行删掉。

---

## 1. 项目是什么

- **项目名称**:`banksys_sy_wangzhihao`
- **一句话目标**:基于银行营销数据,提供交互式数据分析与机器学习预测系统,帮助业务人员洞察客户行为并预测认购意向。
- **使用者/受益者**:银行业务分析师、营销人员;通过数据探索发现规律,通过在线预测辅助营销决策。
- **核心功能**:
  - 数据分析交互页面:对银行营销数据进行多维度可视化探索(分布、相关性、转化漏斗等)。
  - 在线预测系统:基于离线训练的 ML 模型,用户通过点选输入客户特征,实时预测该客户是否会认购定期存款。
- **输入/数据**(如有):
  - 数据来源:葡萄牙银行营销活动数据集(UCI Bank Marketing)
  - 规模:`train.csv`(22501 行,含标签) + `test.csv`(7501 行,含标签)
  - 特征:20 个(含人口统计、营销接触、社会经济指标),目标变量 `subscribe`(yes/no)
  - 是否进 Git:数据文件不进 Git(通过 `.gitignore` 排除);本地开发直接读取 `data/` 目录

## 2. 技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| 语言/运行时 | Python 3.8.5 | 用户指定版本 |
| Web/应用框架 | Streamlit | 纯 Python 构建数据应用,无需前端代码,适合快速交付数据分析与预测界面 |
| 数据处理 | pandas、numpy | 表格数据处理与数值计算 |
| 可视化 | plotly / matplotlib | 交互式图表,适合数据探索 |
| 机器学习 | scikit-learn | 经典分类模型(逻辑回归、随机森林、XGBoost 等),离线训练 + 在线预测 |
| 测试 | pytest | Python 主流测试框架 |
| 格式/静态检查 | ruff | 统一格式与 lint,替代 flake8+isort+black |
| 打包/运行 | Docker | 容器化部署,环境可复现 |
| CI/CD | GitHub Actions | 通用、可视化、适合教学与团队协作 |

## 3. 目录地图

```text
banksys_sy_wangzhihao/
├── standards/                       # AI 项目记忆与通用规范
├── data/                            # 银行营销数据(不进 Git)
│   ├── train.csv                    # 训练集 22501 行
│   └── test.csv                     # 测试集 7501 行
├── app.py                           # Streamlit 应用入口
├── src/                             # 业务逻辑源码
│   ├── __init__.py
│   ├── data_loader.py               # 数据加载与预处理
│   ├── analysis.py                  # 数据分析与可视化逻辑
│   ├── model_trainer.py             # 离线模型训练
│   └── predictor.py                 # 在线预测服务
├── tests/                           # 测试目录
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_analysis.py
│   ├── test_model_trainer.py
│   └── test_predictor.py
├── models/                          # 训练好的模型文件(不进 Git,由训练脚本生成)
├── requirements.txt                 # 生产运行依赖
├── requirements-dev.txt             # 本地/CI 检查依赖
├── Dockerfile                       # 容器构建文件
├── .dockerignore
├── .gitignore
├── .github/workflows/
│   ├── ci.yml
│   └── cd.yml
├── memory/                          # Claude Code 持久记忆
└── README.md
```

> 新增目录前先更新本节,避免项目越做越散。

## 4. 质量门槛

| 类型 | 本项目标准 |
|---|---|
| 格式检查 | `ruff format --check .` |
| 静态检查 | `ruff check .` |
| 单元测试 | `pytest` |
| 覆盖率 | `pytest --cov=src --cov-fail-under=80` |
| 构建 | `docker build` 成功 |
| 业务/模型指标 | 模型 AUC ≥ 0.70、准确率 ≥ 0.75;数据分析页面加载无报错;预测接口响应时间 < 2s |

## 5. 不变约束

- 密钥、密码、私钥、Token **绝不写进代码或文档**,只进 GitHub Secrets / 环境变量。
- 大文件、数据集(`data/*.csv`)、模型产物(`models/*.pkl`)不进 Git,通过 `.gitignore` 排除。
- `main` 分支受保护,日常开发必须走 feature 分支 + PR。
- CI 红灯不合并。
- Streamlit 应用端口固定为 **8888**。
- 容器名称固定为 `banksys_sy_wangzhihao`。

## 6. 部署/CI 占位符取值

> `guides/` 和 workflow 里的通用占位符,在本项目里的真实值只写这里。

| 占位符 | 本项目取值 | 说明 |
|---|---|---|
| `<APP>` | `banksys_sy_wangzhihao` | 应用名/镜像名/容器名 |
| `<DEPLOY_DIR>` | `/opt/banksys_sy_wangzhihao` | 服务器部署目录 |
| `<PORT>` | `8888` | 服务端口 |
| `<PYVER>` | `3.8.5` | Python 版本 |
| `<HEALTHCHECK>` | `/health` 或根路径 `/` | 健康检查地址(Streamlit 无内置 health,需自建或 curl 首页) |
| `<SSH_USER>` | 待配置 | 部署用户,如 `root` 或 `deploy` |
| `<SSH_HOST>` | 待配置 | 服务器公网 IP 或域名 |
