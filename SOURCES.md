# 资料来源与归属

本仓库 `原始资料/` 下的 10 份资料用于演示检索增强问答。以下清单依据 [`data/metadata/documents.json`](data/metadata/documents.json)。

**版权说明**：这些资料的著作权归各自作者或发布方所有，本仓库只为可复现的检索演示而保留其文本快照与图表截图，**不对其授予任何许可**。使用、转载前请查阅原始链接中的条款。本仓库目前没有为任何内容（包括代码）声明开源许可证。

## 文档清单

| document_id | 标题 | 发布方 / 作者 | 原始链接 | 仓库内文件 |
|---|---|---|---|---|
| `lewis-2020-rag` | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | Lewis 等（论文） | https://arxiv.org/abs/2005.11401 | `原始资料/papers/lewis-2020-rag.md` |
| `gao-2024-rag-survey` | Retrieval-Augmented Generation for Large Language Models: A Survey | Gao 等（论文） | https://arxiv.org/abs/2312.10997 | `原始资料/papers/gao-2024-rag-survey.md` |
| `singh-2025-agentic-rag-survey` | Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG | Singh 等（论文） | https://arxiv.org/abs/2501.09136 | `原始资料/papers/singh-2025-agentic-rag-survey.md` |
| `rag-guide` | Retrieval-Augmented Generation (RAG) | Pinecone | https://www.pinecone.io/learn/retrieval-augmented-generation/ | `原始资料/pinecone/rag-guide.md` |
| `chunking-strategies` | Chunking Strategies for LLM Applications | Pinecone | https://www.pinecone.io/learn/chunking-strategies/ | `原始资料/pinecone/chunking-strategies.md` |
| `rerankers-two-stage-retrieval` | Rerankers and Two-Stage Retrieval | Pinecone | https://www.pinecone.io/learn/series/rag/rerankers/ | `原始资料/pinecone/rerankers-two-stage-retrieval.md` |
| `rerank-results` | Rerank results | Pinecone（API 文档） | https://docs.pinecone.io/guides/search/rerank-results | `原始资料/pinecone/rerank-results.md` |
| `retrieval` | Retrieval | LangChain（文档） | https://docs.langchain.com/oss/python/langchain/retrieval.md | `原始资料/langchain/retrieval.md` |
| `semantic-search-knowledge-base` | Build a semantic search engine with LangChain | LangChain（教程） | https://docs.langchain.com/oss/python/langchain/knowledge-base.md | `原始资料/langchain/semantic-search-knowledge-base.md` |
| `agentic-rag` | Build a custom RAG agent with LangGraph | LangChain（教程） | https://docs.langchain.com/oss/python/langgraph/agentic-rag.md | `原始资料/langchain/agentic-rag.md` |

## 文本快照的形式

- 三篇论文的 Markdown 是开发时从 PDF 转换并清洗得到的派生文本；表格等内容可能存在转换误差，以原论文为准。
- Pinecone 与 LangChain 的 Markdown 是网页或官方 Markdown 文档的正文快照，内容可能已被原站更新。
- PDF、HTML 原件不随本仓库发布。

## 图表与图表描述

| 内容 | 位置 | 性质与归属 |
|---|---|---|
| 图表截图 | `原始资料/资料图表/**/*.png` | 从上述原始资料中截取的图表或表格，归原资料作者/发布方所有 |
| 图片说明 | 与 PNG 同名的 `.txt` | 开发时为每张图写的定位与检索说明（来源文件、页码/顺序、主题、关键词），入库时读取 |
| 图表描述（68 条） | `data/metadata/retrieval-eval/experiment-41-final-delivery/release-inputs/visual-descriptions.jsonl` | 由模型对照原图撰写，经人工与 AI 审核修订后验收的文本描述；用于检索和作为回答证据。它们是对第三方图表的描述，不改变原图的归属，描述可能有误，以原图为准 |
| 仅引用图题 id（24 个） | 同目录 `reference-only-text-ids.json` | 开发中的审核决定：这些图题行只供图表描述引用，不作为独立候选参与排序 |
| 来历与哈希 | 同目录 `provenance.json` | 记录上述审核输入的来源与 sha256 |

图表描述和仅引用 id 无法从 Markdown 自动重新生成；重建索引时脚本只核对它们与原文、原图位置的一致性。

## 其他元数据

`data/metadata/` 下的 `source-identities.json`（来源名称与别名）、`figure-captions.json`（图题清单）、`concept-profiles.json` 与 `corpus-terminology/vocabulary-draft.json`（术语词表）是开发过程中根据上述资料整理的辅助数据，用于来源识别和检索，内容所指的原始文本仍归原作者所有。
