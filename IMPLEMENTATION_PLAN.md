# RAG 系统 LangChain + FastAPI 实现规划

## 目录
1. [功能规划](#功能规划)
2. [技术栈规划](#技术栈规划)
3. [项目结构](#项目结构)
4. [模块设计](#模块设计)
5. [API 设计](#api-设计)
6. [实现步骤](#实现步骤)
7. [性能指标](#性能指标)

---

## 功能规划

### 核心功能

#### 1. 文档处理流程
```
用户上传文件（PDF/DOCX）
    ↓
文档解析（使用 manual.py）
    ├─ PDF 处理：OCR → 版面分析 → 表格识别 → 文本合并
    └─ DOCX 处理：层级识别 → 图片提取 → 表格识别
    ↓
文本分块（Token 限制 512）
    ├─ naive_merge：简单合并策略
    └─ hierarchical：保持结构层级
    ↓
向量化（BAAI/bge-large-zh）
    ├─ 批量生成向量
    └─ 存入 ChromaDB
    ↓
完成，可供查询
```

#### 2. 查询工作流
```
用户提问
    ↓
生成查询向量
    ↓
[Redis] 检查缓存
    ├─ 缓存命中 → 直接返回
    └─ 缓存未命中 → 继续
        ↓
[ChromaDB] 向量检索
        ├─ 向量相似度搜索
        └─ 返回 Top-K results
        ↓
[可选] 重排（Reranker）
        └─ 提升结果排序精度
        ↓
[Redis] 缓存结果
        ↓
返回给用户（JSON）
```

#### 3. 支持的功能
- ✓ PDF 文档处理（包含 OCR）
- ✓ DOCX 文档处理（包含图片）
- ✓ 动态分块（可配置 token 数）
- ✓ 中文文本向量化
- ✓ 向量检索
- ✓ 关键词搜索（可选）
- ✓ 结果重排（可选）
- ✓ 查询缓存
- ✓ 异步处理
- ✓ 批量操作

---

## 技术栈规划

### 核心技术

| 模块 | 技术选型 | 说明 | 版本 |
|------|--------|------|------|
| **Web 框架** | FastAPI | 现代异步框架 | 0.104+ |
| **文档解析** | manual.py | 已有实现（PDF/DOCX）| - |
| **文本处理** | LangChain | 链式处理框架 | 0.1+ |
| **向量化** | sentence-transformers | BAAI/bge-large-zh | 2.2+ |
| **向量数据库** | ChromaDB | 轻量级本地向量库 | 0.4+ |
| **缓存** | Redis | 高性能缓存 | 3.0+ |
| **异步任务** | Celery | 后台任务队列 | 5.3+ |
| **ORM** | SQLAlchemy | 数据库 ORM | 2.0+ |
| **数据验证** | Pydantic | 数据验证库 | 2.0+ |
| **日志** | Python logging | 系统日志 | 内置 |

### 依赖库清单

```txt
# Web 框架
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0

# LangChain 相关
langchain==0.1.0
langchain-community==0.0.10

# 向量化
sentence-transformers==2.2.2
torch==2.1.0

# 向量数据库
chromadb==0.4.21

# 缓存
redis==5.0.1

# 异步任务
celery==5.3.4

# 数据库
sqlalchemy==2.0.23
pymysql==1.1.0  # 如果使用 MySQL

# 文档处理（已有）
python-docx==0.8.11
pillow==10.1.0

# 日志和监控
python-multipart==0.0.6

# 开发工具
pytest==7.4.3
pytest-asyncio==0.21.1
```

---

## 项目结构

```
rag_backend_test/
│
├── app/
│   ├── __init__.py
│   ├── main.py                          # FastAPI 应用入口
│   ├── config.py                        # 配置管理
│   │
│   └── api/
│       ├── __init__.py
│       ├── documents.py                 # 文档管理 API
│       │   ├─ POST /api/documents/upload
│       │   ├─ GET /api/documents/{doc_id}
│       │   ├─ DELETE /api/documents/{doc_id}
│       │   └─ GET /api/documents
│       │
│       └── search.py                    # 搜索 API
│           ├─ POST /api/search/query
│           ├─ GET /api/search/history
│           └─ POST /api/search/batch
│
├── core/
│   ├── __init__.py
│   ├── document_processor.py             # 文档处理服务
│   │   ├─ load_document()
│   │   ├─ parse_pdf()
│   │   ├─ parse_docx()
│   │   └─ chunk_text()
│   │
│   ├── embedding_service.py              # 向量生成服务
│   │   ├─ load_model()
│   │   ├─ encode_text()
│   │   ├─ encode_batch()
│   │   └─ get_embeddings()
│   │
│   ├── vector_store.py                   # ChromaDB 向量存储
│   │   ├─ add_documents()
│   │   ├─ search()
│   │   ├─ delete_documents()
│   │   └─ get_collection()
│   │
│   ├── retriever.py                      # 检索服务
│   │   ├─ vector_search()
│   │   ├─ keyword_search()
│   │   ├─ hybrid_search()
│   │   └─ rerank_results()
│   │
│   ├── reranker.py                       # 重排服务
│   │   ├─ load_reranker()
│   │   ├─ rerank()
│   │   └─ score()
│   │
│   └── pipeline.py                       # 完整 RAG 管道
│       ├─ DocumentPipeline
│       ├─ SearchPipeline
│       └─ ChatPipeline
│
├── database/
│   ├── __init__.py
│   ├── base.py                          # 数据库基类
│   ├── models.py                        # 数据模型
│   │   ├─ Document
│   │   ├─ Chunk
│   │   ├─ SearchQuery
│   │   └─ User
│   │
│   └── crud.py                          # 数据库操作
│       ├─ create_document()
│       ├─ update_document()
│       ├─ delete_document()
│       └─ get_chunks()
│
├── schemas/
│   ├── __init__.py
│   ├── document.py                      # 文档相关 schema
│   │   ├─ DocumentUpload
│   │   ├─ DocumentResponse
│   │   └─ DocumentListResponse
│   │
│   └── search.py                        # 搜索相关 schema
│       ├─ SearchQuery
│       ├─ SearchResult
│       └─ SearchResponse
│
├── utils/
│   ├── __init__.py
│   ├── cache.py                         # Redis 缓存管理
│   │   ├─ CacheManager
│   │   ├─ set_cache()
│   │   ├─ get_cache()
│   │   └─ delete_cache()
│   │
│   ├── logging_config.py                # 日志配置
│   ├── constants.py                     # 常量定义
│   │
│   └── helpers.py                       # 辅助函数
│       ├─ generate_id()
│       ├─ normalize_text()
│       └─ calculate_similarity()
│
├── tasks/
│   ├── __init__.py
│   └── document_tasks.py                # 异步任务
│       ├─ process_document_task()
│       ├─ generate_embeddings_task()
│       └─ cleanup_task()
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                      # Pytest 配置
│   ├── test_document_processor.py
│   ├── test_embedding_service.py
│   ├── test_vector_store.py
│   ├── test_retriever.py
│   ├── test_api_documents.py
│   └── test_api_search.py
│
├── .env.example                         # 环境变量示例
├── requirements.txt                     # 依赖清单
├── docker-compose.yml                   # 依赖服务配置
├── Dockerfile                           # 应用 Dockerfile
├── README.md                            # 项目文档
└── IMPLEMENTATION_GUIDE.md              # 实现指南
```

---

## 模块设计

### 1. 文档处理模块（document_processor.py）

**功能**: 解析 PDF/DOCX，分块文本

```python
class DocumentProcessor:
    """文档处理器"""

    def load_document(self, file_path: str) -> bytes:
        """加载文档"""
        pass

    def parse_pdf(self, binary: bytes) -> List[Chunk]:
        """解析 PDF 文档"""
        # 使用 manual.py 的 Pdf 类
        pass

    def parse_docx(self, binary: bytes) -> List[Chunk]:
        """解析 DOCX 文档"""
        # 使用 manual.py 的 Docx 类
        pass

    def chunk_text(self, text: str, chunk_token_num: int = 512) -> List[Chunk]:
        """分块文本"""
        # 支持多种分块策略
        pass
```

**输入**: 文件二进制数据
**输出**: `List[Chunk(text, metadata, source)]`

---

### 2. 向量化模块（embedding_service.py）

**功能**: 将文本转为向量

```python
class EmbeddingService:
    """向量生成服务"""

    def __init__(self, model_name: str = "BAAI/bge-large-zh"):
        """初始化向量模型"""
        pass

    def encode_text(self, text: str) -> List[float]:
        """单个文本转向量"""
        pass

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """批量文本转向量"""
        pass

    def get_model_info(self) -> dict:
        """获取模型信息"""
        pass
```

**输入**: 文本或文本列表
**输出**: `List[float] # 768 维向量`

---

### 3. 向量存储模块（vector_store.py）

**功能**: 使用 ChromaDB 存储和查询向量

```python
class VectorStore:
    """ChromaDB 向量存储"""

    def __init__(self, persist_dir: str = "./chroma_db"):
        """初始化向量存储"""
        pass

    def add_documents(self, documents: List[Document], kb_id: str):
        """添加文档到向量库"""
        # ChromaDB 批量插入
        pass

    def search(self, query_embedding: List[float],
               kb_ids: List[str], top_k: int = 10) -> List[SearchResult]:
        """向量搜索"""
        pass

    def delete_documents(self, doc_ids: List[str]):
        """删除文档"""
        pass

    def get_collection(self, kb_id: str):
        """获取集合"""
        pass
```

**输入**: 向量、文档元数据
**输出**: 搜索结果列表

---

### 4. 检索模块（retriever.py）

**功能**: 混合检索、结果融合

```python
class Retriever:
    """检索器"""

    async def vector_search(self, query_embedding: List[float],
                           kb_ids: List[str], top_k: int) -> List[Result]:
        """向量搜索"""
        pass

    async def keyword_search(self, query: str,
                            kb_ids: List[str], top_k: int) -> List[Result]:
        """关键词搜索（BM25）"""
        pass

    async def hybrid_search(self, query: str,
                           query_embedding: List[float],
                           kb_ids: List[str],
                           top_k: int,
                           weights: Tuple[float, float] = (0.5, 0.5)) -> List[Result]:
        """混合搜索"""
        pass

    async def rerank_results(self, query: str,
                            results: List[Result],
                            reranker) -> List[Result]:
        """重排结果"""
        pass
```

**输入**: 查询向量、查询文本、知识库 ID
**输出**: 排序后的搜索结果

---

### 5. 重排模块（reranker.py）

**功能**: 可选的结果重排

```python
class Reranker:
    """重排器"""

    def __init__(self, model_name: str = "bge-reranker-base"):
        """初始化重排模型"""
        pass

    def rerank(self, query: str, documents: List[str],
              top_k: int = 10) -> List[Tuple[int, float]]:
        """重排文档"""
        # 返回 [(原始索引, 新分数), ...]
        pass
```

**输入**: 查询、候选文档列表
**输出**: 重排后的结果和分数

---

### 6. 缓存模块（cache.py）

**功能**: Redis 缓存管理

```python
class CacheManager:
    """缓存管理器"""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        """初始化 Redis 连接"""
        pass

    def set(self, key: str, value: Any, ttl: int = 300):
        """设置缓存"""
        pass

    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass

    def delete(self, key: str):
        """删除缓存"""
        pass

    def generate_cache_key(self, query: str, kb_ids: List[str],
                          top_k: int) -> str:
        """生成缓存键"""
        pass
```

**输入**: 缓存键、值
**输出**: 存储或检索的数据

---

### 7. 管道模块（pipeline.py）

**功能**: 编排完整的 RAG 流程

```python
class DocumentPipeline:
    """文档处理管道"""

    async def process(self, file: UploadFile, kb_id: str) -> Document:
        """处理上传的文档"""
        # 1. 解析文档
        # 2. 分块
        # 3. 向量化
        # 4. 存入 ChromaDB
        pass

class SearchPipeline:
    """搜索管道"""

    async def execute(self, query: str, kb_ids: List[str],
                     top_k: int = 10, use_reranker: bool = False) -> SearchResponse:
        """执行搜索"""
        # 1. 检查缓存
        # 2. 向量化查询
        # 3. 检索
        # 4. 重排（可选）
        # 5. 缓存结果
        pass
```

---

## API 设计

### 文档管理 API

#### 1. 上传文档
```
POST /api/documents/upload

Request:
{
    "file": <binary>,           # PDF/DOCX 文件
    "kb_id": "kb_001",         # 知识库 ID
    "chunk_token_num": 512      # 可选，分块大小
}

Response:
{
    "code": 0,
    "message": "success",
    "data": {
        "document_id": "doc_abc123",
        "document_name": "product_manual.pdf",
        "status": "processing",
        "chunks_count": 0,
        "created_at": "2025-11-03T10:30:00Z"
    }
}
```

#### 2. 获取文档详情
```
GET /api/documents/{document_id}

Response:
{
    "code": 0,
    "data": {
        "document_id": "doc_abc123",
        "document_name": "product_manual.pdf",
        "kb_id": "kb_001",
        "status": "completed",
        "chunks_count": 125,
        "file_size": 1024000,
        "created_at": "2025-11-03T10:30:00Z",
        "processed_at": "2025-11-03T10:45:00Z"
    }
}
```

#### 3. 列表文档
```
GET /api/documents?kb_id=kb_001&page=1&page_size=10

Response:
{
    "code": 0,
    "data": {
        "documents": [...],
        "total": 50,
        "page": 1,
        "page_size": 10
    }
}
```

#### 4. 删除文档
```
DELETE /api/documents/{document_id}

Response:
{
    "code": 0,
    "message": "success"
}
```

---

### 搜索 API

#### 1. 查询搜索
```
POST /api/search/query

Request:
{
    "query": "什么是 RAG？",
    "kb_ids": ["kb_001", "kb_002"],
    "top_k": 10,
    "use_reranker": false,
    "threshold": 0.5
}

Response:
{
    "code": 0,
    "message": "success",
    "data": {
        "query": "什么是 RAG？",
        "results": [
            {
                "chunk_id": "chunk_123",
                "document_id": "doc_abc123",
                "content": "RAG 是检索增强生成...",
                "similarity": 0.92,
                "page_num": 5,
                "metadata": {
                    "source": "product_manual.pdf",
                    "section": "第一章"
                }
            },
            // ... 最多 10 条
        ],
        "total": 10,
        "response_time_ms": 87,
        "cached": false
    }
}
```

#### 2. 批量查询
```
POST /api/search/batch

Request:
{
    "queries": ["什么是 RAG？", "如何使用 API？"],
    "kb_ids": ["kb_001"],
    "top_k": 5
}

Response:
{
    "code": 0,
    "data": {
        "results": [
            {
                "query": "什么是 RAG？",
                "results": [...]
            },
            {
                "query": "如何使用 API？",
                "results": [...]
            }
        ]
    }
}
```

#### 3. 查询历史
```
GET /api/search/history?limit=20

Response:
{
    "code": 0,
    "data": {
        "history": [
            {
                "query": "什么是 RAG？",
                "kb_ids": ["kb_001"],
                "results_count": 10,
                "timestamp": "2025-11-03T10:30:00Z"
            }
        ]
    }
}
```

---

## 实现步骤

### Phase 1: 基础设施搭建
1. ✓ 创建项目结构
2. ✓ 配置依赖（requirements.txt）
3. ✓ 配置 FastAPI 应用
4. ✓ 配置 ChromaDB 和 Redis
5. ✓ 编写配置管理模块

### Phase 2: 核心服务实现
1. 实现 DocumentProcessor（调用 manual.py）
2. 实现 EmbeddingService（BAAI/bge-large-zh）
3. 实现 VectorStore（ChromaDB）
4. 实现 Retriever（向量检索）
5. 实现 Reranker（可选）

### Phase 3: API 和管道
1. 实现 DocumentPipeline
2. 实现 SearchPipeline
3. 实现文档管理 API
4. 实现搜索 API
5. 实现缓存模块

### Phase 4: 测试和优化
1. 单元测试
2. 集成测试
3. 性能优化
4. 文档完善

---

## 性能指标

### 目标性能

| 操作 | 目标耗时 | 优化策略 |
|------|--------|---------|
| **文档上传** | < 30 秒（50MB） | 异步处理，分块上传 |
| **文档解析** | < 2 分钟（50MB） | GPU 加速 OCR |
| **向量生成** | < 1 分钟（100 chunks） | 批量处理，GPU 加速 |
| **向量搜索** | < 100ms | 索引优化，缓存 |
| **缓存命中** | < 5ms | Redis 本地连接 |
| **重排** | < 200ms | 本地模型，GPU 加速 |
| **总耗时**（无缓存）| < 500ms | 混合优化 |

### 缓存效果

```
假设：100 用户，每小时 1000 查询

无缓存：
  - 每个查询：150ms
  - 总耗时：150 秒
  - CPU：高

有缓存（命中率 40%）：
  - 缓存命中（400）：0.8 秒
  - 缓存未命中（600）：90 秒
  - 总耗时：90.8 秒
  - 改进：39%
```

---

## 后续注意事项

1. **模型选择**
   - BAAI/bge-large-zh：中文最优，768 维
   - 对应的 reranker：bge-reranker-base

2. **向量数据库**
   - ChromaDB：轻量，本地存储
   - 可扩展至 Milvus、Qdrant（生产环境）

3. **缓存策略**
   - Redis TTL：300 秒（可配）
   - 缓存键：MD5(query + kb_ids + top_k)

4. **异步处理**
   - 文档上传：后台任务（Celery）
   - 向量生成：批量异步处理
   - 搜索：FastAPI 异步 API

5. **监控告警**
   - 记录查询耗时、缓存命中率
   - 监控 ChromaDB 和 Redis 状态
   - 设置告警阈值

---

## 相关文件参考

- 详细工作流程: `/home/liudecheng/note_test/WORKFLOW_DETAILED.md`
- manual.py 实现: `/home/liudecheng/rag_flow_test/rag_flow_demo/rag/app/manual.py`

---

**更新时间**: 2025-11-03
**版本**: 1.0