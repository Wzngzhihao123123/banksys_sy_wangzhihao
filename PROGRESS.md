# PROGRESS · 项目进度 〔本项目活记忆 · AI 维护〕

> **作用**:记录当前阶段、已完成、下一步、决策与踩坑。每次会话结束必须更新。
> **格式**:时间倒序,最新在最上面。

---

## 当前状态

- **六步流程阶段**:第 ① 步 — 建仓前准备(文档已就绪,待确认后开工)
- **最后更新**:2026-08-02

---

## 已完成

- (尚无 — 项目初始化阶段)

---

## 第一批 TODO

按六步交付流程排列,每个任务是一个 checklist 项:

### 第①步:建仓 + 配 Secrets
- [ ] 使用 `gh` 创建 GitHub 开源仓库 `banksys_sy_wangzhihao`
- [ ] 初始化仓库:`.gitignore`、占位 `README.md`、标准目录结构
- [ ] ✋ 提示人类配置 GitHub Secrets:`SSH_PRIVATE_KEY` / `SSH_HOST` / `SSH_USER`

### 第②步:开 feature 分支
- [ ] 从 `main` 切出 `feature/1-project-init` 分支
- [ ] ✋ 报分支名,确认后开始开发

### 第③步:模块化开发(逐模块汇报)

**模块 A — 工程骨架**
- [ ] 创建 `src/` 与 `tests/` 目录及 `__init__.py`
- [ ] 编写 `requirements.txt`(streamlit、pandas、numpy、plotly、scikit-learn)
- [ ] 编写 `requirements-dev.txt`(pytest、pytest-cov、ruff)
- [ ] 编写 `Dockerfile`(Python 3.8.5-slim、暴露 8888、启动 streamlit)
- [ ] 编写 `.gitignore`(排除 `data/`、`models/`、`__pycache__`、`.pytest_cache` 等)
- [ ] 编写 `.dockerignore`
- [ ] 编写 `.github/workflows/ci.yml`(ruff format+check + pytest + docker build)
- [ ] 编写 `.github/workflows/cd.yml`(SSH 部署 + docker run + 健康检查)
- [ ] 编写 `app.py` 骨架(侧边栏导航:数据分析 / 在线预测)
- [ ] 编写 `README.md`(项目说明、启动方式、端口)

**模块 B — 数据加载与预处理**
- [ ] 编写 `src/data_loader.py`(加载 CSV、缺失值报告、基础统计)
- [ ] 编写 `tests/test_data_loader.py`

**模块 C — 数据分析页面(US-2)**
- [ ] 编写 `src/analysis.py`(分布图、关系图、认购率分析、相关性热力图)
- [ ] 在 `app.py` 集成数据分析页面(侧边栏筛选器 + 图表联动)
- [ ] 编写 `tests/test_analysis.py`

**模块 D — 模型训练(US-3)**
- [ ] 编写 `src/model_trainer.py`(Pipeline:预处理+训练+评估+保存)
- [ ] 编写 `tests/test_model_trainer.py`
- [ ] 本地运行训练脚本,确认生成 `models/best_model.pkl`
- [ ] 验证 AUC ≥ 0.70、准确率 ≥ 0.75

**模块 E — 在线预测系统(US-4)**
- [ ] 编写 `src/predictor.py`(模型加载、特征预处理、预测)
- [ ] 在 `app.py` 集成预测页面(表单点选 + 结果展示)
- [ ] 编写 `tests/test_predictor.py`

### 第④步:本地 CI 自检
- [ ] 执行 `ruff format --check .` 并修复
- [ ] 执行 `ruff check .` 并修复
- [ ] 执行 `pytest --cov=src --cov-fail-under=80` 并确保全绿
- [ ] ✋ 汇报自检结果

### 第⑤步:触发 PR
- [ ] `git push` 分支
- [ ] `gh pr create` 发起 PR
- [ ] ✋ 报 PR 链接 + CI 状态,停下等待人工审核

### 第⑥步:(人工)审核 → 合并 → CD
- [ ] 人工 Review PR → Merge
- [ ] CD 自动触发 → 盯流水线
- [ ] ✋ 汇报部署结果(端口、健康检查、访问地址)

---

## ADR(Architecture Decision Records · 架构决策记录)

| # | 日期 | 决策 | 理由 | 权衡 |
|---|---|---|---|---|
| 1 | 2026-08-02 | Streamlit 作为应用框架 | 纯 Python,无需前后端分离;内置表单控件适合点选预测;生态成熟 | 不支持多页面独立路由;并发能力有限(适合内部工具) |
| 2 | 2026-08-02 | `duration` 特征训练可用、预测排除 | 通话时长在实际营销前无法知晓;训练时保留用于分析,预测时不作为输入 | 模型 AUC 可能略降,但更符合真实场景 |
| 3 | 2026-08-02 | 模型保存为 pickle + Pipeline | 预处理与模型打包,在线预测只需加载一个文件 | pickle 有版本兼容风险,需记录 Python/scikit-learn 版本 |
| 4 | 2026-08-02 | 数据不进 Git | 数据文件较大(约 3.5MB),且属于外部数据集 | 本地开发需手动放置数据;CI 环境需处理数据缺失(造样本或跳过) |

---

## GOTCHAS(踩坑记录)

- (暂无 — 待开发过程中记录)
