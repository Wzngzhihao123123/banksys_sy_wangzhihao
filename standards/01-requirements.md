# 01 · 需求 / 活 PRD 〔本项目活记忆 · AI 维护〕

> **作用**:这是本项目唯一的需求文档。所有新功能、缺陷、技术债都追加到这里,不要另起多个 PRD 文件。
> **更新时机**:每次有新需求、需求变更、验收标准变化时更新。

---

## 1. 需求来源

| 类型 | 来源 | 进入方式 |
|---|---|---|
| 功能需求 Feature | 用户 / 老师 / 产品 / 客户 | 写成用户故事 |
| 缺陷 Bug | 测试 / 线上日志 / 用户反馈 | 写复现步骤和期望结果 |
| 技术债 Tech Debt | 开发 / Review / CI/CD 故障 | 写影响和修复目标 |

---

## 2. Issue 生命周期

| 阶段 | 状态 | 动作 |
|---|---|---|
| 提出 | Open | 写清场景、目标、验收标准 |
| 排期 | Backlog / Todo | 决定优先级和负责人 |
| 开发 | In Progress | 从 main 开 feature 分支 |
| 评审 | In Review | 提 PR,等待 CI 和 Review |
| 合并 | Done | PR 合并 main,自动关闭 Issue |
| 验收 | Verified | 按验收标准确认 |

**追踪规则**:分支名带 Issue 号,PR 描述写 `closes #<编号>`。

---

## 3. 用户故事模板

```text
### US-<编号> <一句话标题> · 状态: Backlog
作为 <角色>,
我想要 <能力>,
以便 <价值>。

验收标准:
- AC1: Given <前提>,When <动作>,Then <可验证结果>。
- AC2: <补充标准>

技术备注:
- <可选:约束、边界、风险>
```

---

## 4. 需求清单

### US-1 初始化项目工程化与 CI/CD · 状态: Backlog

作为 **项目开发者**,
我想要 项目具备基础工程结构、测试、CI 与 CD 流水线,
以便 后续每次开发都能自动检查代码质量并自动部署。

验收标准:
- AC1: Given 空仓库,When 执行初始化,Then 项目包含标准目录结构(`src/`、`tests/`、`standards/`、`.github/workflows/`)、`requirements.txt`、`requirements-dev.txt`、`Dockerfile`、`.gitignore`。
- AC2: Given 代码提交到 feature 分支,When 发起 PR,Then CI 自动运行 `ruff format --check` + `ruff check` + `pytest --cov=src --cov-fail-under=80`,全部通过。
- AC3: Given CI 全绿且人工合并到 main,When CD 触发,Then 自动构建 Docker 镜像、部署容器并完成健康检查。
- AC4: Given 部署完成,When 访问 `http://<SSH_HOST>:8888`,Then Streamlit 应用正常加载。
- AC5: Given CI/CD 流水线跑通,When 完成,Then `PROGRESS.md` 已更新当前状态。

技术备注:
- 端口 8888;容器名 `banksys_sy_wangzhihao`
- 数据文件(`data/*.csv`)和模型文件(`models/*.pkl`)不进 Git
- 本地开发不强制 Docker,CI 环节负责 `docker build` 验证

---

### US-2 数据分析交互页面 · 状态: Backlog

作为 **银行业务分析师**,
我想要 在 Web 页面上交互式探索银行营销数据,
以便 快速理解数据分布、发现特征规律,为营销策略提供数据支撑。

验收标准:
- AC1: Given 应用启动,When 打开首页,Then 显示数据分析页面,包含数据概览(行数、列数、缺失值统计)。
- AC2: Given 数据概览区,When 选择某个特征列,Then 展示该特征的分布图(数值型:直方图/箱线图;分类型:柱状图)。
- AC3: Given 特征分析区,When 选择两个特征,Then 展示两者关系图(数值-数值:散点图;类别-数值:分组箱线图;类别-类别:堆叠柱状图/热力图)。
- AC4: Given 目标变量 `subscribe`,When 查看认购率分析,Then 展示整体认购率、按关键特征(如 job/marital/education)分组的认购率对比。
- AC5: Given 相关性分析区,When 查看数值特征相关性,Then 展示相关性热力图。
- AC6: Given 数据筛选区,When 用户通过侧边栏按特征值筛选数据,Then 所有图表联动更新,只展示筛选后数据。
- AC7: Given 任意操作,When 页面加载或交互,Then 无报错、图表正常渲染,响应时间 < 3 秒。

技术备注:
- 使用 Streamlit + plotly 实现
- 数据从 `data/train.csv` 加载(本地路径)
- 页面布局:侧边栏放筛选器与导航,主区域放图表

---

### US-3 离线模型训练 · 状态: Backlog

作为 **数据科学家**,
我想要 基于历史营销数据离线训练一个二分类模型,
以便 获得一个可用于在线预测的模型文件,预测客户是否会认购定期存款。

验收标准:
- AC1: Given `data/train.csv`,When 执行训练脚本,Then 完成数据加载、缺失值处理、特征工程(编码分类变量、标准化数值变量)。
- AC2: Given 预处理后数据,When 划分训练/验证集,Then 按 80/20 分层划分,保持正负样本比例一致。
- AC3: Given 训练集,When 训练模型(至少尝试逻辑回归 + 随机森林),Then 输出模型在验证集上的 AUC、准确率、精确率、召回率、F1 分数。
- AC4: Given 训练完成,When 评估模型,Then 选择性能最优模型,AUC ≥ 0.70,准确率 ≥ 0.75。
- AC5: Given 最优模型,When 保存,Then 模型文件保存至 `models/best_model.pkl`,附带特征列表和预处理流水线。
- AC6: Given 训练脚本,When 运行 `pytest tests/test_model_trainer.py`,Then 训练流程可被单元测试覆盖(使用采样数据或 mock,避免耗时过长)。

技术备注:
- 使用 scikit-learn Pipeline 封装预处理+模型,便于在线预测时复用
- 分类变量使用 OrdinalEncoder 或 OneHotEncoder
- 需注意 `duration` 特征(通话时长)在真实预测场景中未知,训练时可作为分析参考但预测时不依赖
- 训练脚本可独立运行:`python -m src.model_trainer`

---

### US-4 在线预测系统 · 状态: Backlog

作为 **银行营销人员**,
我想要 在 Web 页面上通过点选输入客户特征,
以便 系统实时返回该客户是否会认购定期存款的预测结果,辅助营销决策。

验收标准:
- AC1: Given 用户进入预测页面,When 查看输入表单,Then 以点选控件(下拉框、单选框、滑块)展示所有预测所需特征,分为三组:人口统计(age/job/marital/education)、财务(default/housing/loan)、营销接触(contact/month/day_of_week/campaign/pdays/previous/poutcome)及经济指标(emp_var_rate/cons_price_index/cons_conf_index/lending_rate3m/nr_employed)。
- AC2: Given 填写表单,When 所有必填字段已填,Then "预测"按钮变为可点击状态;若有未填字段则提示用户补齐。
- AC3: Given 点击"预测",When 模型已加载,Then 在 2 秒内返回预测结果:显示"会认购"或"不会认购",并附带预测概率置信度。
- AC4: Given 预测完成,When 查看结果区域,Then 同时展示该客户的关键特征与历史数据中相似客户的统计对比(如"您的年龄在历史认购者中处于前 30%")。
- AC5: Given 模型文件不存在,When 点击预测,Then 显示友好错误提示"模型尚未训练,请先运行训练脚本"。
- AC6: Given 用户首次使用,When 打开预测页面,Then 每个输入项旁有简短说明(如 tooltip),帮助理解特征含义。
- AC7: Given 预测页面,When 运行 `pytest tests/test_predictor.py`,Then 预测逻辑(特征预处理、模型加载、输出格式)被单元测试覆盖。

技术备注:
- 使用 Streamlit 表单控件:`st.selectbox`、`st.radio`、`st.slider`、`st.number_input`
- 模型加载使用 `@st.cache_resource` 缓存,避免每次预测重复加载
- 预测时排除 `duration` 特征(通话时长在实际营销前无法预知)
- 预测页面与数据分析页面在同一个 Streamlit 应用中,通过侧边栏导航切换

---

## 5. 非功能需求

- **安全**:密钥只进 Secrets,不进 Git;数据文件和模型文件不进 Git。
- **可维护**:一需求一小 PR(每个 US 一个分支),避免大爆炸式提交;模块间接口清晰。
- **可测试**:核心逻辑(数据加载、特征工程、模型训练、预测)必须有单元测试;覆盖率 ≥ 80%。
- **可部署**:Docker 容器化;部署后健康检查通过;端口 8888。
- **可用性**:页面加载 < 3 秒;预测响应 < 2 秒;首次使用无需额外配置。
- **兼容性**:Python 3.8.5;支持主流浏览器(Chrome/Firefox/Edge)。
