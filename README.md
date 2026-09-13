# RAG-Demo

用一句中文问题，检索本地 10 份 RAG 相关资料，得到**带引用编号、原文位置和原图**的回答。

这是一个在固定资料集上完成安装、重建索引和真实提问验证的演示项目，不是通用问答产品：它只回答这 10 份资料里的内容。现在提供本机网页工作台，也保留命令行入口和 Markdown 答案导出。

## 它能做什么

- **本机网页工作台**：提问和历史、独立的引用证据层、模型设置。安装模型并重建索引后，可双击启动；页面不需要额外构建。
- **中文口语提问**：可以指定资料（“Lewis 那篇论文里……”），也可以不指定。指定了库里没有的资料时，会给出说明并停下，不会擅自改查其他资料。
- **图文证据**：命中图表时，把原图发给生成模型；网页在证据层展示原图，命令行导出的答案中嵌入原图。
- **可追溯**：检索计划、候选、重排、送入模型的证据、生成回执和答案都保存在输出目录，方便核对。
- **名次/索引读数复核**：答案里出现“第 N 位”这类说法、且证据中有位次列或索引代码时，最多追加 3 次调用，要求给出可核对的计数起点和换算证明；其他问题不触发，不额外花钱。它只覆盖这一类表述，**不是全答案的事实核验**。

## 处理流程

```
中文问题
 → DeepSeek：理解需求、识别指定来源、生成英文检索请求
 → GME 向量检索 + BM25 词法补充（每个请求每类证据最多 20 个候选）
 → BGE v2 m3 重排
 → 组装证据（正文 + 图表描述 + 原图）
 → DeepSeek 生成带 [S编号] 引用的回答
 → 名次/索引复核（仅在触发时）
 → 导出答案
```

## 使用的模型

| 用途 | 模型 | 运行位置 |
|---|---|---|
| 向量检索 | `Alibaba-NLP/gme-Qwen2-VL-2B-Instruct`（固定版本） | 本地 GPU |
| 重排 | `BAAI/bge-reranker-v2-m3`（固定版本） | 本地 GPU |
| 需求理解、生成回答 | DeepSeek `deepseek-flash`（OpenAI 兼容接口） | 远程 API，按量付费 |

固定的模型版本见 [`config/runtime.json`](config/runtime.json)，生成参数见 [`config/generation.deepseek-rag.json`](config/generation.deepseek-rag.json)。以上是本项目验证时使用的配置，模型服务后续的可用性和价格以提供方为准。

## 资料（10 份）

| 类别 | 资料 |
|---|---|
| 论文（3） | Lewis 等 *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*；Gao 等 *Retrieval-Augmented Generation for Large Language Models: A Survey*；Singh 等 *Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG* |
| Pinecone（4） | *Retrieval-Augmented Generation (RAG)*；*Chunking Strategies for LLM Applications*；*Rerankers and Two-Stage Retrieval*；*Rerank results* |
| LangChain（3） | *Retrieval*；*Build a semantic search engine with LangChain*；*Build a custom RAG agent with LangGraph* |

原文链接、图表描述的来历和归属见 [SOURCES.md](SOURCES.md)。**这些资料属于各自的作者和发布方，本仓库不对其重新授权。**

## 环境要求

| 项 | 要求 |
|---|---|
| 系统 | Windows x64（唯一验证过的平台；锁定依赖是该平台的包） |
| Python | 3.12（验证版本 3.12.14），**新建**虚拟环境 |
| GPU | NVIDIA，CUDA 12.8，显存 8 GB（验证卡 RTX 5060 Ti 8 GB；GME 与 BGE 依次加载） |
| Node.js | 24.x（只运行 `scripts/*.mjs` 提示词桥接，不需要 `npm install`） |
| 磁盘 | 建议预留 25 GB 以上（模型、PyTorch、缓存和索引；参考值，不是测得的最低要求） |
| 网络 | 安装与下载模型时需要；重建索引可离线；提问时调用 DeepSeek API |
| API 密钥 | DeepSeek API key；网页发送问题或测试连接、命令行加 `--execute` 时使用 |

## 快速开始

在 PowerShell 中执行，示例目录为 `D:\RAG-Demo`。每一步的检查点、网络与缓存边界见 [安装与复现](doc/安装与复现.md)。

### 1. 获取代码并新建环境

```powershell
git clone https://github.com/Jointan-Yixiao/RAG-Demo.git D:\RAG-Demo
cd D:\RAG-Demo
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-runtime-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

### 2. 下载固定版本的模型，设置 BGE 目录

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\_rebuild_runtime.py models --download --bge-dir models\bge-reranker-v2-m3
$env:RAG_BGE_MODEL_DIR = (Resolve-Path models\bge-reranker-v2-m3).Path
```

GME 下载到 Hugging Face 缓存；BGE 的 6 个文件下载到 `models\bge-reranker-v2-m3` 并逐个核对哈希。**`RAG_BGE_MODEL_DIR` 只从当前进程环境读取，不读 `.env`；每开一个新的 PowerShell 窗口都要重新设置。**

### 3. 从资料文本重建索引（不调用付费 API）

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\_rebuild_runtime.py records --out data\runtime\records
.\.venv\Scripts\python.exe -X utf8 scripts\_rebuild_runtime.py encode  --records data\runtime\records --out data\runtime\index
.\.venv\Scripts\python.exe -X utf8 scripts\_rebuild_runtime.py verify  --index data\runtime\index
```

### 4. 打开网页工作台

完成前面环境、模型和索引准备后，双击项目目录中的 **启动工作台.cmd**。浏览器会打开 `http://127.0.0.1:8765`。在设置页配置 API 密钥，或沿用项目 `.env` 中的 `DEEPSEEK_API_KEY`。

- 网页自动识别 `models/bge-reranker-v2-m3`；使用其他模型目录时，启动前设置 `RAG_BGE_MODEL_DIR`。
- 在主界面发送问题会调用真实模型并产生费用；浏览历史和证据不调用模型。设置页的“测试连接”会发送一条小请求。
- 主界面阅读答案，“引用来源 / 证据层”中核对正文、原图和来源；当前每题独立，不自动引用历史问答。
- 关闭页面不会停止后台服务；用完双击 **停止工作台.cmd**。服务只监听本机，不开放局域网访问。

详细说明见 [本地工作台接入与使用](doc/UI-本地工作台接入.md)。

### 5. 使用命令行提问（可选）

```powershell
Copy-Item .env.example .env    # 然后在 .env 中填写 DEEPSEEK_API_KEY=
```

**先预检（不加 `--execute`）**：不读密钥、不联网、不收费。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\_rag_e2e.py --index-dir data\runtime\index --query "向量都找到最像的内容了，为什么还要再排一次？" --out-dir runs\q001-check
```

**确认无误后加 `--execute`**：调用 DeepSeek（**产生费用**），并加载 GME 与 BGE。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\_rag_e2e.py --index-dir data\runtime\index --query "向量都找到最像的内容了，为什么还要再排一次？" --out-dir runs\q001 --execute
```

答案在 `runs\q001\07-answers\`。

- **始终传 `--index-dir data\runtime\index`。** 不传时脚本会寻找仓库中不存在的旧版索引，预检报 `index_error`。
- **每次真实运行使用新的 `--out-dir`。** 已执行过的目录会被拒绝，避免重复收费；中断后先查看已有回执，不要删除记录后盲目重跑。
- **批量提问**用 `--queries-file` 代替 `--query`，每批最多 32 题。示例：

  ```powershell
  .\.venv\Scripts\python.exe -X utf8 scripts\_rag_e2e.py --index-dir data\runtime\index --queries-file examples\questions.json --out-dir runs\examples-check
  ```

  [`examples/questions.json`](examples/questions.json) 中有一题故意指定库里没有的论文；加上 `--execute` 实际执行时，该题会得到“来源不可用”的说明，批次总状态为 `partial`。仅预检不会执行这些问题。
- 输出目录结构和状态含义见 [实时端到端运行说明](doc/RAG-实时端到端运行说明.md)。

## 当前命令行入口

| 命令 | 作用 |
|---|---|
| `scripts/_rag_e2e.py` | 提问：单题 `--query` 或批量 `--queries-file`；不加 `--execute` 只预检 |
| `scripts/_rebuild_runtime.py` | `models` 下载/核对模型；`records` 重建记录；`encode` 编码索引；`verify` 校验索引 |
| `scripts/_answer_semantics.py` | 对已有运行结果单独做名次/索引复核（写入新目录；不加 `--execute` 只做筛查） |
| `scripts/_prepare_delivery.py check` | 按发布清单核对文件哈希，并检查运行模块导入和 `.mjs` 语法 |

默认检索参数：`hybrid`、K=10、候选上限 20；名次复核默认开启（`--answer-review on`）。

## 验证情况

详见 [验证结果](doc/验证结果.md)，简要如下：

- **网页接入验收**：一道真实图文问题完成后台全流程，验证答案、引用、原图、历史及设置接口；自动浏览器被 Chrome 屏蔽，实际点击与视觉验收仍待完成。详见 [工作台验收记录](doc/UI-本地工作台接入.md#验收记录)。
- **E41 复现验收**：同一台电脑的新目录、新 Python 环境中完成安装、重建与 GPU 编码；108 组已保存查询的检索候选与原索引一致；6 道真实提问均有用户可见输出。这是复现验收，**不是新的全量准确率**。
- **E40 端到端回归**：47 道可回答题中候选完整覆盖 45/47、严格端到端通过 44/47；3 道部分通过的质量问题仍保留，见 [WO-008](doc/WO-008-E40端到端完整性与稳定性遗留.md)。
- 测试题围绕现有语料设计，并由独立审查检查结果；不是随机用户流量，不能推算任意问题的准确率，也不代表答案 100% 正确。

**测试**：本仓库携带 43 项运行链路与工作台测试，不需要 GPU 或付费 API：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

其中依赖旧版分层索引的 1 项会跳过；本地没有 GME 分词器缓存时，从原文重建记录的 1 项也会跳过。验证报告中的 752 项是开发项目的完整测试集，**不随本仓库提供，在本仓库中无法运行**。

## 已知限制

- 只验证过 Windows x64 + Python 3.12 + CUDA 12.8 + 8 GB 显存。复现验收在同一台电脑上完成，GME 复用了同版本的本机模型缓存，**尚未在另一台电脑、空缓存条件下完整验证**。
- 只适用于这 10 份资料。68 条图表描述是事先撰写并审核过的输入，不能从 Markdown 自动生成；**换成自己的资料不会自动可用**，需要重新入库并撰写、审核图表描述，这不在本仓库范围内。
- E40 保留的质量问题：只检索图时漏掉正文要点、证据已有但回答遗漏机制、同一题两次运行召回不一致；另有发送非目标图片过多、个别次要表述不准。答案需要结合引用人工核对。
- 部分已部署代码仍位于 `data/metadata/retrieval-eval/experiment-*` 目录（历史位置遗留，不影响运行）。

## 文档

- [本地工作台接入与使用](doc/UI-本地工作台接入.md)：启动停止、提问、证据、设置与已知边界
- [前端接口与行为](doc/UI-frontend.md)：页面路由、交互和待办

- [安装与复现](doc/安装与复现.md)：完整步骤、索引如何从文本重建、网络与缓存边界
- [实时端到端运行说明](doc/RAG-实时端到端运行说明.md)：参数、输出目录、状态含义
- [验证结果](doc/验证结果.md)：各轮指标的口径与局限
- [E40 端到端修复与全量回归](doc/E40-端到端修复与全量回归.md)
- [WO-008 遗留问题](doc/WO-008-E40端到端完整性与稳定性遗留.md)
- [SOURCES.md](SOURCES.md)：资料来源与归属
