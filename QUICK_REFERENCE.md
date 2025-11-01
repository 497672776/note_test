# RAGFlow 快速参考指南

## 核心文件位置速查

### RAG 核心算法
```
分块算法      → /rag/nlp/__init__.py (naive_merge, hierarchical_merge, tree_merge)
分词系统      → /rag/nlp/rag_tokenizer.py (混合分词 + Trie树)
混合搜索      → /rag/nlp/search.py (Dealer 类)
词项权重      → /rag/nlp/term_weight.py (IDF + NER + 词性)
查询处理      → /rag/nlp/query.py (FulltextQueryer)
```

### 嵌入和重排
```
嵌入模型      → /rag/llm/embedding_model.py (20+ 种服务)
重排模型      → /rag/llm/rerank_model.py (13+ 种服务)
聊天模型      → /rag/llm/chat_model.py
```

### 数据库和存储
```
向量DB连接    → /rag/utils/doc_store_conn.py (DocStoreConnection 抽象)
Elasticsearch → /rag/utils/es_conn.py
Infinity      → /rag/utils/infinity_conn.py
S3/OSS存储    → /rag/utils/s3_conn.py, /rag/utils/oss_conn.py
ORM模型       → /api/db/db_models.py (Peewee)
数据库服务    → /api/db/services/ (各业务逻辑)
```

### 知识图谱
```
图搜索        → /graphrag/search.py (KGSearch 类)
图提取        → /graphrag/general/graph_extractor.py
图嵌入        → /graphrag/general/entity_embedding.py (Node2Vec)
社区检测      → /graphrag/general/leiden.py
```

---

## 关键类和函数速查

### 1. 分块 (Chunking)

```python
# 简单分块（推荐用于大多数场景）
from rag.nlp import naive_merge
chunks = naive_merge(
    sections="文本内容",
    chunk_token_num=512,           # token数限制
    delimiter="\n。；！？",         # 分隔符
    overlapped_percent=20          # 20% 重叠
)

# 层级分块（适合结构化文档）
from rag.nlp import hierarchical_merge
chunks = hierarchical_merge(
    bull=0,        # bullet pattern 类型 (0-4)
    sections=[...],
    depth=2        # 层级深度
)
```

### 2. 分词 (Tokenization)

```python
from rag.nlp import rag_tokenizer

# 基本分词
tokens = rag_tokenizer.tokenize("自然语言处理技术")
# 输出: "自然 语言 处理 技术"

# 细粒度分词
fine_tokens = rag_tokenizer.fine_grained_tokenize(tokens)

# 词项权重计算
from rag.nlp import term_weight
dealer = term_weight.Dealer()
weights = dealer.weights(["深度", "学习"])
# 返回: [("深度", 0.45), ("学习", 0.55)]
```

### 3. 嵌入 (Embedding)

```python
# 获取嵌入模型
from rag.llm.embedding_model import EmbeddingFactory

embed = EmbeddingFactory.get_embed(
    "OpenAI",
    key="sk-xxx",
    model_name="text-embedding-3-small"
)

# 编码文本
vectors, tokens = embed.encode(["文本1", "文本2"])
# vectors: shape (2, 1536)

# 编码查询
query_vec, tokens = embed.encode_queries("查询文本")
```

### 4. 混合搜索 (Hybrid Search)

```python
from rag.nlp.search import Dealer
from rag.utils.doc_store_conn import FusionExpr

dealer = Dealer(dataStore)

# 执行搜索
result = dealer.search(
    req={
        "question": "查询内容",
        "topk": 10,
        "similarity": 0.1,
        "page": 1,
        "size": 5
    },
    idx_names=["index_name"],
    kb_ids=["kb_id"],
    emb_mdl=embed_model,
    rank_feature={PAGERANK_FLD: 10}
)

# 结果结构
print(result.ids)           # chunk ID列表
print(result.field)         # chunk 详细内容
print(result.keywords)      # 提取的关键词
print(result.aggregation)   # 按文档的聚合统计
```

### 5. 重排 (Reranking)

```python
from rag.llm.rerank_model import RerankerFactory

rerank = RerankerFactory.get_rerank(
    "Cohere",
    key="api_key",
    model_name="rerank-english-v3.0"
)

# 对搜索结果重排
scores, tokens = rerank.similarity(
    query="查询内容",
    texts=["候选1", "候选2", "候选3"]
)
# scores: numpy array of shape (3,)

# 在检索流程中使用
sim, tksim, vtsim = dealer.rerank_by_model(
    rerank_mdl=rerank,
    sres=search_result,
    query="查询",
    tkweight=0.3,
    vtweight=0.7
)
```

### 6. 知识图谱 (Knowledge Graph)

```python
from graphrag.search import KGSearch

kg_search = KGSearch(dataStore)

# 查询重写
type_keywords, entities = kg_search.query_rewrite(
    llm=chat_model,
    question="谁与张三合作过？",
    idxnms=["index_name"],
    kb_ids=["kb_id"]
)

# 实体搜索
entities = kg_search.get_relevant_ents_by_keywords(
    keywords=["公司", "科技"],
    filters={"kb_ids": ["kb_id"]},
    idxnms=["index_name"],
    kb_ids=["kb_id"],
    emb_mdl=embed_model,
    sim_thr=0.3
)

# 关系搜索
relations = kg_search.get_relevant_relations_by_txt(
    txt="合作协议",
    filters={"kb_ids": ["kb_id"]},
    idxnms=["index_name"],
    kb_ids=["kb_id"],
    emb_mdl=embed_model
)
```

---

## 常用配置参数

### 分块参数
| 参数 | 默认值 | 范围 | 说明 |
|------|-------|------|------|
| chunk_token_num | 512 | 128-2048 | chunk大小（token） |
| overlapped_percent | 0 | 0-50 | 重叠比例(%) |
| delimiters | "\n。；！？" | - | 分隔符列表 |

### 搜索参数
| 参数 | 默认值 | 说明 |
|------|-------|------|
| topk | 1024 | 初始候选数 |
| similarity | 0.1 | 向量相似度阈值 |
| min_match | 0.3 | 全文匹配阈值 |
| page | 1 | 页码 |
| size | topk | 每页大小 |

### 嵌入参数
| 参数 | 说明 |
|------|------|
| batch_size | 16（OpenAI）、4（QWen）、1（ZhipuAI） |
| max_tokens | 8191（OpenAI）、2048（QWen）、3072（Zhipu） |
| input_type | "document" 或 "query"（Cohere风格） |

### 重排参数
| 参数 | 说明 |
|------|------|
| topk | 返回数量 |
| tkweight | token权重（0.0-1.0） |
| vtweight | 向量权重（0.0-1.0） |
| rank_feature | 排名特征字典 |

---

## 常见问题速解

### Q1: 选择哪种分块算法？
**A:** 
- 结构化文档（论文、法律文件）→ `hierarchical_merge`
- 非结构化文本（新闻、博客）→ `naive_merge`
- 纯文本流 → `naive_merge` + 大的 `overlapped_percent`

### Q2: 嵌入模型选择？
**A:**
- **最佳质量**：OpenAI text-embedding-3-large (1536维)
- **性价比**：HuggingFace TEI + BAAI/bge-large (768维)
- **本地部署**：Ollama + nomic-embed-text
- **中文优化**：通义千问、BAAI/bge-m3

### Q3: 如何提高检索准确率？
**A:**
1. 调整 `chunk_token_num`（较小更精细）
2. 降低 `similarity` 阈值（从0.1降至0.05）
3. 增加 `overlapped_percent`（20-30%）
4. 使用重排模型（提升15-30%）
5. 配置排名特征（融入PageRank）

### Q4: 向量DB选择？
**A:**
- **Elasticsearch**：成熟稳定，全文搜索好
- **Infinity**：向量查询性能好，自动归一化
- **选择标准**：<1000 chunks 用 ES，>1000 用 Infinity

### Q5: 知识图谱何时使用？
**A:**
- 有明确实体类型和关系 → 使用KG-RAG
- 只有文本内容 → 使用混合RAG
- 需要多跳推理 → 必须使用KG-RAG

---

## 性能优化Checklist

- [ ] 启用批处理（16并发嵌入）
- [ ] 配置连接池（max_connections=10）
- [ ] 启用Redis缓存
- [ ] 调整RERANK_LIMIT=64
- [ ] 设置合理的相似度阈值
- [ ] 使用Tag特征进行个性化排序
- [ ] 定期更新PageRank
- [ ] 监控向量DB的索引大小

---

## 调试技巧

### 查看中间结果
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看分词结果
logging.debug(f"Tokens: {rag_tokenizer.tokenize(text)}")

# 查看权重计算
logging.debug(f"Weights: {dealer.tw.weights(tokens)}")

# 查看搜索结果
logging.debug(f"Search results: {result.field}")
```

### 性能分析
```python
import time

start = time.time()
# 执行搜索
result = dealer.search(...)
print(f"Search time: {time.time() - start:.2f}s")
```

---

## 生产部署清单

### 基础设施
- [ ] PostgreSQL 13+ 配置
- [ ] Elasticsearch/Infinity 集群
- [ ] Redis 缓存
- [ ] S3/OSS 对象存储

### 模型服务
- [ ] 嵌入模型（OpenAI/Jina API 或本地TEI）
- [ ] 重排模型（可选，按需）
- [ ] LLM 服务（用于图提取和查询重写）

### 监控和日志
- [ ] 检索质量监控（MRR、NDCG）
- [ ] 系统性能监控（延迟、吞吐）
- [ ] 错误日志和告警
- [ ] Token 成本追踪

### 数据管理
- [ ] 定期备份向量索引
- [ ] 定期更新统计信息（IDF、PageRank）
- [ ] 数据清理和归档策略

