# RAG 实时端到端运行说明

更新：2026-09-13（E41 后）。本文说明提问入口 `scripts/_rag_e2e.py` 的参数、输出目录和状态含义。安装和重建索引见 [安装与复现](安装与复现.md)；质量结论见 [验证结果](验证结果.md)、[E40 报告](E40-端到端修复与全量回归.md) 和 [WO-008](WO-008-E40端到端完整性与稳定性遗留.md)。

从一句原始中文问题可以运行完整链路，并保存检索计划、候选、重排、证据、生成回执、名次复核回执和带原文/原图的答案。资料无法使用时（例如指定了库里没有的文档），也会导出面向用户的说明。

## 运行

在仓库根目录（示例 `D:\RAG-Demo`）下执行，先设置 BGE 模型目录（每个新 PowerShell 窗口都要设置）：

```powershell
$env:RAG_BGE_MODEL_DIR = (Resolve-Path models\bge-reranker-v2-m3).Path
```

默认只预检：不读密钥、不联网、不收费。

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\_rag_e2e.py --index-dir data\runtime\index --query "向量都找到最像的内容了，为什么还要再排一次？" --out-dir runs\live-question-001-check
```

检查通过后，加 `--execute` 真实运行（调用 DeepSeek，产生费用）：

```powershell
.\.venv\Scripts\python.exe -X utf8 scripts\_rag_e2e.py --index-dir data\runtime\index --query "向量都找到最像的内容了，为什么还要再排一次？" --out-dir runs\live-question-001 --execute
```

- 始终传 `--index-dir data\runtime\index`；不传时会寻找仓库中没有的旧版索引。
- 每次真实运行使用一个未执行过的输出目录；已有 `run-reservation.json` 的目录会拒绝重复运行，以免误收费。
- 中断后先查看已有回执。入口不会自动恢复或重复发出付费请求，不要删除记录后盲目重跑。

## 批量输入

```json
{"queries":[{"id":"q1","original_query":"用户原话"},{"id":"q2","original_query":"另一个完整问题"}]}
```

用 `--queries-file` 代替 `--query`，**每批最多 32 题**，超过会在入口校验时拒绝，需自行分批。示例文件：[`examples/questions.json`](../examples/questions.json)。`--workers 3` 只让最终生成并发；前置处理逐题隔离，不把同批其他问题当成上下文。`DEEPSEEK_API_KEY` 由适配器从环境变量或仓库根目录 `.env` 中读取，不要写进问题文件或命令。

## 默认行为

| 阶段 | 默认 | 说明 |
|---|---|---|
| 检索 | `--retrieval-mode hybrid --k 10 --lexical-cap 20` | 每个子请求、每种证据类型先取向量前 10，再用 BM25 词法补充。**20 是该组总候选上限（含向量的 10 个），不是再额外加 20 个。** 向量候选及其顺序不变，来源和证据类型过滤不变。`--retrieval-mode vector` 可切回纯向量检索作对照 |
| 索引 | `--index-dir data\runtime\index` | 由 `scripts/_rebuild_runtime.py` 从文本重建的平铺索引 |
| 重排 | 原生 BAAI/bge-reranker-v2-m3 | 超长输入报错，不静默截断 |
| 图片 | `--image-policy all_available` | 候选带入的图片都会发送，可能包含用户没问的图 |
| 名次复核 | `--answer-review on` | 见下节 |

## 名次/索引复核（06b）

只在问题涉及位次/索引、证据中有位次列或索引代码、且答案出现“第 N 位”一类自然名次表述时触发；未触发的题 0 额外调用。触发后**每题最多 3 次额外调用**（复核、至多一次修订、修订后复核）。放行只看脚本对复核模型所给证明的核验（原文引句、列名、计数起点与换算），不看模型自己说的 pass。

- 复核与修订写独立回执到 `06b-answer-review`，原 `06-generation` 回执保持只读；答案导出只采用已批准且回执哈希一致的版本。
- 它**只检查名次/索引这一类表述，不是通用的全答案事实核验**。没有触发或通过复核，都不代表答案其他内容正确。
- `--answer-review off` 可关闭，用于对照。
- 对已有运行结果单独复核：`scripts\_answer_semantics.py --contexts <运行目录>\05-context\contexts.json --generation-dir <运行目录>\06-generation --out-dir <新目录>`，不加 `--execute` 只做筛查。

## 结果位置

| 位置 | 内容 |
|---|---|
| `run-report.json` | 总体状态、每题状态、各阶段参数与用量（含检索模式、词法补充数、复核额外调用） |
| `01-frontend` | 需求判断、英译、词库与来源处理、决策审计和调用记录 |
| `02-embedding` 至 `04-rerank` | 本轮查询向量、检索候选（含 `lexical-supplement.json` 及每个候选的向量/词法名次）和实际 BGE 排序 |
| `05-context` | 实际用于生成的问题和证据 |
| `06-generation` | 请求身份、实际发送的图片、usage 和原始回答；不保存推理正文 |
| `06b-answer-review` | 名次复核的筛查结果、复核/修订回执和证明核验 |
| `07-answers` | 带引用和原图的 Markdown 答案及结构化汇总；未生成答案的题也有 Markdown 说明和 `.user-response.json` |

## 状态怎么读

- `answered`：生成成功，且已知的前置检查没有缺口。**这不是语义正确认证**；有依据地说明“证据里没有这个数字”也属于此状态。
- `answered_with_frontend_gaps`：生成成功，但前置有子问题或来源告警，答案页会显示提示。
- `failed`：处理错误、记录不足、缺少有效回执等，需要查看实际原因。
- `no_answer_source_unresolved`：用户指定的资料未找到或无法唯一确定。`no_answer_source_ambiguous`：有明确歧义依据。两者都**不检索、不调用生成**，输出会说明原因，并给出补充资料、明确文档或明确放宽来源的下一步；系统不会自行改查全库。
- `no_answer_no_evidence`：实际检索过但没有候选；不会被误报为来源缺失。传输或格式错误也不会伪装成资料缺失。

批次中同时有正常回答和被阻塞的题时，总状态为 `partial`，进程返回 1；全部为可解释的无答案时可为 `no_answer`。请结合每题状态判断。例如 E41 的 6 题验收批次中有 1 题是预期的来源阻塞，总状态就是 `partial`，其余 5 题完成生成，6 题都有用户可见输出。

真正的资料缺口仍需结合原始问题、候选和答案审查；前置分类器的偏差需要查看 `01-frontend` 中的审计记录，总体状态不能覆盖所有情况。

## 历史说明

早期轮次（E37–E39）使用纯向量 K=10、无名次复核的版本，其指标反映当时的实现，不代表当前默认。这些轮次的报告和原始运行记录保存在开发项目中，未随本仓库发布。
