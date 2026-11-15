# EEG 智库 Agent

面向脑电（EEG）研究的知识库与智能工作台。项目将文献导入、可追溯检索问答和受控的 EEG 分析工具结合，帮助研究者整理资料、查阅实验设置与理解已有分析结果。

## 界面预览

以下为项目开发与验收期间的实际截图：桌面端截图拍摄于 2026 年 11 月 5–6 日，移动端截图拍摄于 11 月 12 日。展示数据和能力随本地配置而变化。

### 研究工作台

桌面端工作台展示对话模式、研究内容、结果及单试次分析入口。

![桌面研究工作台](docs/screenshots/desktop-workbench.png)

### 资料上传与知识库

上传 PDF、DOCX 或 Markdown，跟踪资料解析、索引与入库状态。

![资料上传与知识库](docs/screenshots/desktop-import.jpg)

### Skills 模型与分析能力

查看模型与分析能力、适用条件及启用状态。11 月 12 日版本还增加了预处理 Skills，见下方移动端截图。

![模型与分析 Skills](docs/screenshots/desktop-skills.jpg)

### 用户登录

通过研究账号进入个人研究空间。

![用户登录](docs/screenshots/desktop-login.png)

### 模型与 API 设置

管理模型连接和服务配置。

![模型与 API 设置](docs/screenshots/desktop-model-settings.jpg)

### 移动端：样本登记、Agent 分析与预处理

11 月 12 日验收截图展示脑电样本登记、模型分析和预处理能力。

| 脑电数据登记 | Agent 模型分析 | 预处理 Skills |
| --- | --- | --- |
| ![脑电数据登记](docs/screenshots/eeg-registration.jpg) | ![Agent 模型分析](docs/screenshots/agent-inference.jpg) | ![预处理 Skills](docs/screenshots/preprocessing-skills.jpg) |

## 功能

- **资料导入**：支持 PDF、DOCX 和 Markdown，保留原件，解析并切分内容，建立可检索索引；PDF 解析可配置 MinerU。
- **知识问答**：结合向量检索、HyDE、RRF 融合与可选重排，组织证据并展示来源。
- **研究 Agent**：通过任务规划与工具调用处理研究问题，支持任务状态、过程和结果查看。
- **EEG 工具**：提供单试次数据与模型相关分析能力，具体可用性依赖本地样本、权重及运行环境；特征提取结果不等同于临床诊断或自动分类结论。
- **研究空间**：支持账号、会话、模型连接以及资料管理。

## 架构

```text
浏览器工作台
  ├─ 资料导入 API → 解析 / 切分 → Embedding → Milvus
  │                         └─ 原件与图片 → MinIO
  └─ 查询与 Agent API → 检索 / 融合 / 重排 → 证据回答与工具执行
                            └─ MongoDB：历史与任务状态
```

主要技术包括 Python、FastAPI、LangGraph、Milvus、MongoDB、MinIO、BGE 和 MNE。实际启用的模型和服务取决于配置。

## 本地运行

需要 Python 3.11 或更高版本，以及可访问的 Milvus、MongoDB、MinIO、模型服务和相应模型文件。依赖体积较大，模型和数据不包含在仓库中。

```powershell
# 在仓库目录执行；需预先安装 uv
uv sync
Copy-Item .env.example .env
# 编辑 .env，填写模型服务地址、密钥和数据库连接
.venv\Scripts\python.exe -m uvicorn app.api.http.import_server:app --host 127.0.0.1 --port 8000
# 另开一个终端
.venv\Scripts\python.exe -m uvicorn app.api.http.query_server:app --host 127.0.0.1 --port 8001
```

打开 `http://127.0.0.1:8001/html` 进入研究工作台，`http://127.0.0.1:8000/html` 进入资料导入。首次使用需要完成账号设置。模型下载、服务连接、MinerU 等需按 `.env.example` 和实际环境配置；仅启动页面不代表所有检索或分析能力均已就绪。

可选的运动想象预处理依赖及使用说明见 [docs/mi-preprocessing.md](docs/mi-preprocessing.md)。本仓库不包含原项目机器专用的 Hyper-V 配置、数据卷或数据库。

## 目录

```text
app/api/        HTTP API 与请求模型
app/agent/      Agent 编排、工具和 EEG 技能
app/process/    导入和查询流程
app/rag/        文档处理、检索和回答服务
app/resources/  页面与提示词
app/workspace/  研究空间与连接管理
tests/          离线回归测试
docs/           使用说明与界面截图
```

本项目用于研究与工程探索，不作为医疗诊断依据。
