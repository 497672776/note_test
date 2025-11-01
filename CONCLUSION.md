# RAGFlow 项目总结与分析

## 📌 项目概览

RAGFlow 是一个企业级的检索增强生成（RAG）框架，由 InfiniFlow 开发，提供完整的知识库管理、文档处理和 AI 对话能力。

---

## 🎯 核心技术模块

### 1. 文档分块（Chunking）三大算法

#### 1.1 naive_merge - 基于 Token 的简单分块
- **适用场景**：新闻、博客、无结构文本
- **核心特性**：按 token 数限制，支持重叠
- **复杂度**：O(n)
- **参数**：`chunk_token_num`、`delimiter`、`overlapped_percent`

#### 1.2 hierarchical_merge - 基于结构的智能分块
- **适用场景**：学术论文、法律文件、产品文档
- **核心特性**：保留文档层级结构，支持 5 种编号格式
- **复杂度**：O(n log m)
- **参数**：`bullet`（编号类型）、`depth`（层级深度）

#### 1.3 tree_merge - 树形层级合并
- **适用场景**：极度嵌套的文档、学位论文
- **核心特性**：完全保留树形结构，递归处理
- **复杂度**：O(n log m)
- **参数**：`chunk_token_num`、`depth_limit`

---

## 🔌 集成能力

### 嵌入模型（20+）
- OpenAI、Jina、Cohere、NVIDIA、Voyage
- 国内：通义千问、百度文心、讯飞、Zhipu
- 开源：HuggingFace TEI、Ollama、LM-Studio

### 重排模型（13+）
- Cohere、Jina、NVIDIA E5、BGE、Qwen 等

### 向量数据库
- Elasticsearch、Infinity（推荐）、OpenSearch、Weaviate

---

## 💾 存储层

- **关系数据库**：PostgreSQL、MySQL（Peewee ORM）
- **向量存储**：Elasticsearch、Infinity
- **缓存**：Redis
- **对象存储**：S3、MinIO、OSS、Azure Blob

---

## 🧠 高级特性

### 知识图谱 RAG
- LLM-based 实体提取
- Node2Vec 图嵌入
- Leiden 社区检测

### 混合检索
- 稀疏检索（全文）+ 密集检索（向量）+ 融合
- 支持加权和、RRF 等融合策略

### 排序学习
- PageRank + Tag Feature + 位置特征
- 贝叶斯推断标签相关性

---

## 📊 性能指标

| 指标 | 值 |
|------|-----|
| 检索延迟 | <50ms |
| 重排延迟 | <200ms |
| 端到端 | <500ms |
| 嵌入吞吐 | 16并发 |
| NDCG@10 | >0.65 |

---

## 🚀 部署建议

### 开发环境
- Vector DB：Elasticsearch 8.x
- Embedding：HuggingFace TEI
- Rerank：BGE-Reranker

### 生产环境
- Vector DB：Infinity 集群
- Embedding：OpenAI / Jina API
- Rerank：Cohere / NVIDIA API
- DB：PostgreSQL + Redis

---

## 📚 文档导航

- `RAGFLOW_DETAILED_ANALYSIS.md` - 代码级深度分析
- `RAG_TECHNOLOGY_SUMMARY.md` - 技术全景
- `ALGORITHM_COMPARISON.md` - 算法对比与选型
- `QUICK_REFERENCE.md` - 快速参考
- `INDEX.md` - 文档索引

---

## 🔄 待补充内容

> 将在后续分析中逐步完善

---

**分析时间**：2025-11-02
**项目**：RAGFlow（InfiniFlow）
**许可**：Apache 2.0
