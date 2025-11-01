# RAGFlow 项目深入技术分析

## 项目概览
RAGFlow 是由 InfiniFlow 开发的现代 RAG（检索增强生成）框架，提供完整的知识库管理、文档处理和 AI 对话能力。项目采用模块化架构，包含多个核心组件。

---

## 1. RAG 相关算法

### 1.1 文档分块（Chunking）策略

#### 策略类型
项目支持多种分块算法：
- **基础分块 (naive_merge)**：基于令牌数和分隔符的简单分块
- **层级分块 (hierarchical_merge)**：基于文档结构的智能分块
- **树形合并 (tree_merge)**：基于文档层级关系的分块

#### 核心实现 (`/rag/nlp/__init__.py` - naive_merge函数)

```python
def naive_merge(sections: str | list, chunk_token_num=128, delimiter="\n。；！？", overlapped_percent=0):
    """
    Token-based chunk merging algorithm
    
    Parameters:
    - chunk_token_num: 目标块大小（token数）
    - delimiter: 分隔符（支持多个）
    - overlapped_percent: 块重叠比例（0-100）
    
    Algorithm Flow:
    1. Parse delimiters and extract regex pattern
    2. Iterate through sections
    3. Split sections by delimiters
    4. Merge subsections while respecting token limit
    5. Apply overlap if specified
    """
    cks = [""]
    tk_nums = [0]
    
    def add_chunk(t, pos):
        tnum = num_tokens_from_string(t)
        # 当前chunk超过token限制时，创建新chunk
        if cks[-1] == "" or tk_nums[-1] > chunk_token_num * (100 - overlapped_percent)/100.:
            if cks:
                # 应用重叠：取前一个chunk的末尾
                overlapped = RAGFlowPdfParser.remove_tag(cks[-1])
                t = overlapped[int(len(overlapped)*(100-overlapped_percent)/100.):] + t
            cks.append(t)
            tk_nums.append(tnum)
        else:
            cks[-1] += t
            tk_nums[-1] += tnum
```

**关键特性**：
- 支持可配置的重叠比例（避免信息丢失）
- 基于 token 计数的自适应分块（而非固定字符数）
- 支持多个自定义分隔符

#### 层级分块实现 (hierarchical_merge)

```python
def hierarchical_merge(bull, sections, depth):
    """
    基于文档结构的智能分块
    
    Algorithm:
    1. 检测文档的bullet/heading模式（5种语言/格式）
    2. 为每个section分配层级（level）
    3. 按层级进行二分查找，构建chunk边界
    4. 合并相邻低层级内容（token-aware）
    
    Bullet Pattern Types:
    - 中文编号：第N章/节/条、（N）
    - 阿拉伯编号：N. 、N.M、N.M.L等
    - 中文数字：第一、二、三等
    - 英文：PART、CHAPTER、SECTION、ARTICLE
    - Markdown：#、##、###等
    """
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [[] for _ in range(bullets_size + 2)]
    
    # 为每个section分配层级
    for i, (txt, layout) in enumerate(sections):
        for j, p in enumerate(BULLET_PATTERN[bull]):
            if re.match(p, txt.strip()):
                levels[j].append(i)
                break
        else:
            if re.search(r"(title|head)", layout):
                levels[bullets_size].append(i)
            else:
                levels[bullets_size + 1].append(i)
    
    # 二分查找和chunk构建
    cks = []
    for i, arr in enumerate(levels[:depth]):
        for j in arr:
            if readed[j]:
                continue
            cks.append([j])
            # 收集该层级下的所有后续内容
            for ii in range(i + 1, len(levels)):
                jj = binary_search(levels[ii], j)
                if jj >= 0:
                    cks[-1].append(levels[ii][jj])
```

**分块场景**：
- 论文、书籍：利用章节结构
- 法律文件：利用条款编号
- 网页：利用heading层级

### 1.2 向量嵌入（Embedding）实现

#### 支持的嵌入模型

项目支持 **20+ 种嵌入服务**（`/rag/llm/embedding_model.py`）：

| 服务商 | 模型示例 | 特点 |
|--------|---------|------|
| OpenAI | text-embedding-ada-002 | 批量大小≤16 |
| HuggingFace | TEI | 本地部署，自动截断 |
| 通义千问 | text_embedding_v2 | 批量大小≤4，支持查询/文档类型区分 |
| Jina | jina-embeddings-v3 | 支持长上下文 |
| Cohere | embed-english-v3.0 | 支持不同输入类型 |
| NVIDIA | NV-Embed-QA | 专用QA嵌入 |
| Bedrock | Titan、Cohere | AWS集成 |
| 本地模型 | 通过Ollama | 离线运行 |

#### 嵌入调用流程

```python
class BuiltinEmbed(Base):
    def encode(self, texts: list):
        batch_size = 16
        ress = None
        token_count = 0
        
        # 批处理以避免速率限制
        for i in range(0, len(texts), batch_size):
            embeddings, token_count_delta = self._model.encode(texts[i : i + batch_size])
            token_count += token_count_delta
            if ress is None:
                ress = embeddings
            else:
                ress = np.concatenate((ress, embeddings), axis=0)
        return ress, token_count
    
    def encode_queries(self, text: str):
        # 查询向量通常有特殊处理（如Cohere的input_type="search_query"）
        return self._model.encode_queries(text)
```

**关键特性**：
- 自动批处理（避免API超限）
- Token 计数和成本追踪
- 查询/文档区分化嵌入
- 自动文本截断

### 1.3 向量检索和相似度计算

#### 混合检索架构

项目实现了 **三层混合检索** (`/rag/nlp/search.py`）：

```python
class Dealer:
    def search(self, req, idx_names, kb_ids, emb_mdl=None, ...):
        """
        混合检索实现
        
        Retrieval Pipeline:
        1. 全文匹配（Sparse）：基于token匹配
        2. 向量匹配（Dense）：基于语义相似度
        3. 融合（Fusion）：组合两种方法的结果
        """
        
        # Step 1: 全文检索
        matchText, keywords = self.qryr.question(qst, min_match=0.3)
        
        # Step 2: 向量检索
        if emb_mdl is not None:
            matchDense = self.get_vector(qst, emb_mdl, topk, similarity=0.1)
            
            # Step 3: 融合（RRF-style weighted combination）
            fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
            matchExprs = [matchText, matchDense, fusionExpr]
        
        res = self.dataStore.search(src, highlightFields, filters, matchExprs, 
                                   orderBy, offset, limit, idx_names, kb_ids)
```

#### 相似度计算算法

```python
def hybrid_similarity(self, avec, bvecs, atks, btkss, tkweight=0.3, vtweight=0.7):
    """
    混合相似度 = 向量相似度 × 权重1 + 词项相似度 × 权重2
    
    Parameters:
    - avec: 查询向量
    - bvecs: 候选向量列表
    - atks: 查询token列表
    - btkss: 候选token列表
    - tkweight: 词项权重（默认0.3）
    - vtweight: 向量权重（默认0.7）
    
    Algorithm:
    1. 计算余弦相似度
    2. 计算token-level相似度
    3. 加权组合
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    sims = CosineSimilarity([avec], bvecs)  # (1, N)
    tksim = self.token_similarity(atks, btkss)  # List[float]
    
    if np.sum(sims[0]) == 0:
        return np.array(tksim), tksim, sims[0]
    
    # 加权融合
    return (np.array(sims[0]) * vtweight + 
            np.array(tksim) * tkweight), tksim, sims[0]
```

#### 融合策略（Fusion）

项目支持多种融合方法：

```python
# 定义：加权和融合
fusionExpr = FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})
# 权重配置：5% 全文分数 + 95% 向量分数

# 倒数排名融合（RRF）可用但需通过向量DB实现
# Infinity vs Elasticsearch差异：
# - Elasticsearch：不会在融合前归一化，需手动rerank
# - Infinity：自动归一化每种方法的分数
```

### 1.4 重排（Reranking）算法

#### 支持的重排模型

项目集成了 **13+ 个重排服务**（`/rag/llm/rerank_model.py`）：

| 服务 | 特点 |
|------|------|
| Jina | 多语言，支持长文本 |
| Cohere | 向量感知 |
| NVIDIA | 专用QA和检索重排 |
| Qwen | 通义千问系列 |
| BGE-Reranker | 开源，可本地部署 |
| Voyage | 高精度 |

#### 重排实现流程

```python
def rerank_by_model(self, rerank_mdl, sres, query, tkweight=0.3, ...):
    """
    使用专门的重排模型进行重新排序
    
    Pipeline:
    1. 提取候选chunk的内容
    2. 使用重排模型计算相关性分数
    3. 结合token相似度和向量相似度
    4. 按综合分数排序
    """
    
    # 构建输入
    ins_tw = []
    for i in sres.ids:
        content_ltks = sres.field[i][cfield].split()
        title_tks = sres.field[i].get("title_tks", "").split()
        important_kwd = sres.field[i].get("important_kwd", [])
        tks = content_ltks + title_tks + important_kwd
        ins_tw.append(tks)
    
    # 调用重排模型
    tksim = self.qryr.token_similarity(keywords, ins_tw)
    vtsim, _ = rerank_mdl.similarity(query, 
                    [remove_redundant_spaces(" ".join(tks)) for tks in ins_tw])
    
    # 综合分数 = token权重 × (token相似度 + rank_feature) + 向量权重 × 向量分数
    return tkweight * (np.array(tksim) + rank_fea) + vtweight * vtsim, tksim, vtsim
```

### 1.5 融合（Fusion）和聚合策略

#### 融合方法

```python
# 1. 加权和融合
FusionExpr("weighted_sum", topk, {"weights": "0.05,0.95"})

# 2. 倒数排名融合（RRF）通过向量DB实现
# 公式：score = Σ(1/(k+rank_i)) 其中k通常为60

# 3. 组织级聚合（Tag-based Aggregation）
rank_fea = self._rank_feature_scores(query_rfea, search_res)
# 利用文档标签特征进行个性化排序
```

#### 聚合实现

```python
def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, ...):
    """
    完整的检索管道
    
    1. 初始检索：hybrid search
    2. 重排：使用重排模型或tag特征
    3. 分页：应用RERANK_LIMIT进行中间排序
    4. 聚合：按文档名聚合统计
    """
    
    # 计算重排限制（确保是page_size的倍数）
    RERANK_LIMIT = math.ceil(64/page_size) * page_size
    
    # 执行初始检索
    sres = self.search(req, [index_name(tid) for tid in tenant_ids],
                      kb_ids, embd_mdl, rank_feature=rank_feature)
    
    # 重排
    if rerank_mdl and sres.total > 0:
        sim = self.rerank_by_model(rerank_mdl, sres, question, ...)
    else:
        sim = self.rerank(sres, question, ...)
    
    # 分页和聚合
    ranks = {"chunks": [], "doc_aggs": {}}
    for i in idx:
        chunk = sres.field[sres.ids[i]]
        dnm = chunk.get("docnm_kwd", "")
        
        ranks["chunks"].append(d)
        if dnm not in ranks["doc_aggs"]:
            ranks["doc_aggs"][dnm] = {"doc_id": did, "count": 0}
        ranks["doc_aggs"][dnm]["count"] += 1
```

---

## 2. NLP 和文本处理算法

### 2.1 分词（Tokenization）策略

#### 多语言分词架构

项目实现了 **混合分词系统**（`/rag/nlp/rag_tokenizer.py`）：

```python
class RagTokenizer:
    def tokenize(self, line):
        """
        混合分词流程
        
        Pipeline:
        1. 预处理：全角转半角、繁体转简体
        2. 语言识别：中/英文分割
        3. 中文分词：使用Trie树的前向-后向动态规划
        4. 英文处理：Porter Stemming + WordNet Lemmatization
        5. 细粒度分词：处理复合词
        """
        
        # Step 1: 规范化
        line = self._strQ2B(line).lower()  # 全角→半角
        line = self._tradi2simp(line)      # 繁→简
        
        # Step 2: 语言分割
        arr = self._split_by_lang(line)
        
        res = []
        for L, lang in arr:
            if not lang:  # 英文
                res.extend([self.stemmer.stem(self.lemmatizer.lemmatize(t)) 
                           for t in word_tokenize(L)])
            else:  # 中文
                # Step 3: 前向最大匹配
                tks, s = self.maxForward_(L)
                # Step 4: 后向最大匹配
                tks1, s1 = self.maxBackward_(L)
                
                # Step 5: 比较两种结果并选择最佳分词
                # 使用DFS动态规划处理歧义
                tkslist = []
                self.dfs_(L, 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))
        
        return self.merge_(res)
```

#### Trie树构建

```python
def __init__(self):
    trie_file_name = self.DIR_ + ".txt.trie"
    
    if os.path.exists(trie_file_name):
        self.trie_ = datrie.Trie.load(trie_file_name)
    else:
        self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(self.DIR_ + ".txt")  # 从词表文件构建
        self.trie_.save(trie_file_name)

def loadDict_(self, fnm):
    """
    词表格式：word frequency pos_tag
    例：
    中华人民共和国 9.5 nt
    """
    with open(fnm, "r", encoding='utf-8') as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            freq = float(parts[1])
            pos_tag = parts[2] if len(parts) > 2 else ""
            
            F = int(math.log(freq / self.DENOMINATOR) + 0.5)
            self.trie_[self.key_(word)] = (F, pos_tag)
```

#### 细粒度分词

```python
def fine_grained_tokenize(self, tks):
    """
    进一步分解复合词
    
    例：
    "自然语言处理" → ["自然", "语言", "处理"]
    "deep learning" → ["deep", "learning"]
    """
    tks = tks.split()
    zh_num = len([1 for c in tks if c and is_chinese(c[0])])
    
    # 如果主要是英文，按'/'分割
    if zh_num < len(tks) * 0.2:
        res = []
        for tk in tks:
            res.extend(tk.split("/"))
        return " ".join(res)
    
    # 中文处理：DFS分词并选择最佳分割
    res = []
    for tk in tks:
        if len(tk) < 3:
            res.append(tk)
            continue
        
        tkslist = []
        self.dfs_(tk, 0, [], tkslist)
        stk = self.sortTks_(tkslist)[1][0]  # 取第二好的分割
        res.append(" ".join(stk))
    
    return " ".join(self.english_normalize_(res))
```

### 2.2 文本预处理

#### 预处理流程

```python
def pretoken(self, txt, num=False, stpwd=True):
    """
    查询/chunk预处理
    
    Steps:
    1. 移除特殊字符
    2. 分词
    3. 去停用词
    4. 过滤纯数字token
    """
    
    patt = [r"[~—\t @#%!<>,\.\?\":;'...]+"]
    
    res = []
    for t in rag_tokenizer.tokenize(txt).split():
        tk = t
        
        # 去停用词
        if (stpwd and tk in self.stop_words) or (
                re.match(r"[0-9]$", tk) and not num):
            continue
        
        # 特殊字符处理
        for p in patt:
            if re.match(p, t):
                tk = "#"
                break
        
        if tk != "#" and tk:
            res.append(tk)
    
    return res
```

### 2.3 关键信息提取和权重计算

#### 词项权重（Term Weight）算法

```python
def weights(self, tks, preprocess=True):
    """
    综合词项权重计算：结合IDF、词性、实体类型等
    
    权重公式：
    w = IDF_combined × NER_factor × POS_factor
    其中 IDF_combined = 0.3*IDF_freq + 0.7*IDF_df
    """
    
    def idf(s, N):
        return math.log10(10 + ((N - s + 0.5) / (s + 0.5)))
    
    def ner(t):
        # 数值类：权重×2
        if num_pattern.match(t):
            return 2
        # 短字母：权重×0.01
        if short_letter_pattern.match(t):
            return 0.01
        # 实体识别权重
        if t in self.ne:
            m = {"toxic": 2, "func": 1, "corp": 3, "loca": 3, "sch": 3}
            return m[self.ne[t]]
        return 1
    
    def postag(t):
        # 词性权重
        tag = rag_tokenizer.tag(t)
        if tag in {"r", "c", "d"}:  # 代词、连接词、副词
            return 0.3
        if tag in {"ns", "nt"}:  # 地点、机构
            return 3
        if tag == "n":  # 名词
            return 2
        return 1
    
    tw = []
    for tk in tks:
        freq_score = freq(tk)
        df_score = df(tk)
        
        idf1 = idf(freq_score, 10000000)
        idf2 = idf(df_score, 1000000000)
        
        # 综合权重
        w = (0.3 * idf1 + 0.7 * idf2) * ner(tk) * postag(tk)
        tw.append((tk, w))
    
    # 归一化
    S = np.sum([s for _, s in tw])
    return [(t, s / S) for t, s in tw]
```

### 2.4 去重和聚类

#### 内容去重

```python
# Citation Insertion（引用插入）时的去重
def insert_citations(self, answer, chunks, chunk_v, embd_mdl, 
                     tkweight=0.1, vtweight=0.9):
    """
    为生成的答案插入引用并去重
    
    去重策略：
    1. 按句子分割答案
    2. 对每个句子计算与chunk的混合相似度
    3. 若相似度高于阈值，则插入引用
    4. 避免重复引用（同一chunk最多引用4次）
    """
    
    pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
    
    for i, t in enumerate(pieces):
        if len(t) < 5:
            continue
        
        # 计算相似度
        sim = self.qryr.hybrid_similarity(ans_v[i], chunk_v,
                                         tokens_i, chunks_tks,
                                         tkweight, vtweight)
        
        mx = np.max(sim) * 0.99
        if mx < threshold:
            continue
        
        # 收集候选引用
        cites[i] = list(set([str(ii) for ii in range(len(chunk_v)) 
                           if sim[ii] > mx]))[:4]  # 最多4个
```

---

## 3. 数据库技术

### 3.1 支持的数据库系统

项目支持 **多种数据库后端**：

#### 向量数据库
- **Elasticsearch**：传统全文搜索 + 向量搜索
- **Infinity**：优化的向量检索（自动分数归一化）
- **OpenSearch**：AWS兼容版本

#### 文档存储
- **PostgreSQL / MySQL**：元数据和关系数据
- **Peewee ORM**：数据库抽象层

#### 其他存储
- **Redis**：缓存和会话
- **MinIO / OSS / S3**：对象存储

### 3.2 数据库架构

#### ORM 模型

```python
# /api/db/db_models.py
from peewee import Model, CharField, IntegerField, TextField

class JSONField(LongTextField):
    """自定义JSON字段"""
    def db_value(self, value):
        if value is None:
            value = self.default_value
        return json_dumps(value)
    
    def python_value(self, value):
        if not value:
            return self.default_value
        return json_loads(value)

class ListField(JSONField):
    """列表字段"""
    default_value = []

class SerializedField(LongTextField):
    """支持Pickle和JSON序列化"""
    def __init__(self, serialized_type=SerializedType.PICKLE, **kwargs):
        self._serialized_type = serialized_type
        super().__init__(**kwargs)
    
    def db_value(self, value):
        if self._serialized_type == SerializedType.PICKLE:
            return serialize_b64(value, to_str=True)
        elif self._serialized_type == SerializedType.JSON:
            return json_dumps(value)
```

#### 连接池管理

```python
from playhouse.pool import PooledMySQLDatabase, PooledPostgresqlDatabase

# 使用连接池改善并发性能
db = PooledMySQLDatabase(
    'database_name',
    user='user',
    password='password',
    host='localhost',
    max_connections=10  # 连接池大小
)
```

### 3.3 向量检索实现

#### 向量相似度距离类型

```python
class MatchDenseExpr:
    def __init__(self,
        vector_column_name: str,
        embedding_data: list | np.ndarray,
        embedding_data_type: str,      # "float"
        distance_type: str,             # "cosine", "l2", "ip"
        topk: int = 10,
        extra_options: dict = {}
    ):
        # extra_options = {"similarity": 0.1}  # 相似度阈值
```

支持的距离类型：
- **cosine**：余弦相似度（推荐，与嵌入模型契合）
- **l2**：欧几里得距离
- **ip**：内积

#### 混合查询实现

```python
class Dealer:
    def search(self, req, idx_names, kb_ids, emb_mdl=None, ...):
        """混合查询的核心实现"""
        
        # 1. 获取向量
        if emb_mdl is None:
            matchExprs = [matchText]
        else:
            matchDense = self.get_vector(qst, emb_mdl, topk, 
                                        req.get("similarity", 0.1))
            
            # 2. 配置融合
            fusionExpr = FusionExpr("weighted_sum", topk, 
                                   {"weights": "0.05,0.95"})
            matchExprs = [matchText, matchDense, fusionExpr]
        
        # 3. 执行检索
        res = self.dataStore.search(
            selectFields=src,
            highlightFields=highlightFields,
            condition=filters,
            matchExprs=matchExprs,    # 多表达式
            orderBy=orderBy,
            offset=offset,
            limit=limit,
            index_names=idx_names,
            kb_ids=kb_ids,
            rank_feature=rank_feature  # 排名特征
        )
```

### 3.4 索引和查询优化

#### 索引策略

```python
# 全文索引字段（支持高亮）
class Chunk:
    content_ltks = TextField()      # 分词后的内容（可全文搜索）
    title_tks = TextField()         # 标题分词
    important_kwd = ListField()     # 重要关键词（提高权重）
    
    # 向量索引
    q_1536_vec = VectorField()      # 1536维向量（OpenAI）
    q_384_vec = VectorField()       # 384维向量（小模型）

# 索引创建
def createIdx(self, indexName: str, knowledgebaseId: str, vectorSize: int):
    """
    创建索引时配置：
    1. 分词器：中英文混合分词
    2. 向量索引：HNSW或IVF算法
    3. 字段权重：title权重 > important_kwd > content
    """
```

#### 查询优化技巧

```python
# 1. 相似度阈值调整（自动降级）
if total == 0:
    # 降低min_match阈值重试
    matchText, _ = self.qryr.question(qst, min_match=0.1)
    matchDense.extra_options["similarity"] = 0.17
    res = self.dataStore.search(...)

# 2. 中间重排（RERANK_LIMIT）
RERANK_LIMIT = math.ceil(64/page_size) * page_size
# 先取RERANK_LIMIT个候选，再进行分页和重排

# 3. 排名特征（Rank Feature）
rank_feature = {PAGERANK_FLD: 10}
# 结合PageRank等图算法优化排序
```

---

## 4. 机器学习相关

### 4.1 排序学习（Learning to Rank）

#### 排名特征工程

```python
def _rank_feature_scores(self, query_rfea, search_res):
    """
    计算排名特征分数，用于提升相关结果
    
    Features:
    1. PageRank：文档全局重要性
    2. Tag Feature：文档标签相关性
    3. 位置特征（Position）
    
    Scoring:
    score = Σ(query_feature[i] × doc_feature[i])
    """
    
    rank_fea = []
    pageranks = []
    
    # 提取PageRank分数
    for chunk_id in search_res.ids:
        pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
    
    pageranks = np.array(pageranks, dtype=float)
    
    if not query_rfea:
        return pageranks
    
    # 计算查询特征归一化分母
    q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() 
                             if t != PAGERANK_FLD]))
    
    for chunk_id in search_res.ids:
        tag_fea = eval(search_res.field[chunk_id].get(TAG_FLD, "{}"))
        
        # 向量点积
        nor = 0
        denor = 0
        for t, sc in tag_fea.items():
            if t in query_rfea:
                nor += query_rfea[t] * sc
            denor += sc * sc
        
        if denor == 0:
            rank_fea.append(0)
        else:
            # 余弦相似度
            rank_fea.append(nor / np.sqrt(denor) / q_denor)
    
    # 加权组合 = 特征相似度×10 + PageRank
    return np.array(rank_fea) * 10. + pageranks
```

#### 标签特征提取

```python
def tag_content(self, tenant_id, kb_ids, doc, all_tags, topn_tags=3, ...):
    """
    为文档提取和计算标签特征
    
    Algorithm:
    1. 使用文档内容进行全文查询
    2. 统计结果中的标签分布
    3. 使用贝叶斯推断计算特征权重
    
    Feature Weight:
    weight = P(tag|content) × relevance_score
           = (count + 1) / (total + smoothing) / P(tag)
    """
    
    match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"],
                                   doc.get("important_kwd", []),
                                   keywords_topn=30)
    
    res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 
                               0, 0, idx_nm, kb_ids, ["tag_kwd"])
    aggs = self.dataStore.getAggregation(res, "tag_kwd")
    
    if not aggs:
        return False
    
    cnt = np.sum([c for _, c in aggs])
    # BM25-style: (count + 1) / (total + S) / prior_probability
    tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / 
                                max(1e-6, all_tags.get(a, 0.0001))))
                     for a, c in aggs],
                    key=lambda x: x[1] * -1)[:topn_tags]
    
    doc[TAG_FLD] = {a.replace(".", "_"): c for a, c in tag_fea if c > 0}
    return True
```

### 4.2 特征工程

#### 多层级特征提取

```python
def insert_citations(self, answer, chunks, chunk_v, embd_mdl, 
                    tkweight=0.1, vtweight=0.9):
    """
    构造用于答案-chunk匹配的特征
    
    Features:
    1. 文本特征：答案句子的分词
    2. 向量特征：答案句子的嵌入
    3. 权重特征：关键词权重
    """
    
    # 按句子分割
    pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
    
    # 仅保留有意义的句子
    idx = []
    pieces_ = []
    for i, t in enumerate(pieces):
        if len(t) < 5:
            continue
        idx.append(i)
        pieces_.append(t)
    
    if not pieces_:
        return answer, set([])
    
    # 获取句子向量
    ans_v, _ = embd_mdl.encode(pieces_)
    
    # 获取句子token
    chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                  for ck in chunks]
    
    # 计算混合相似度
    sim = self.qryr.hybrid_similarity(ans_v[i], chunk_v,
                                     rag_tokenizer.tokenize(pieces_[i]).split(),
                                     chunks_tks,
                                     tkweight=0.1, vtweight=0.9)
```

### 4.3 模型评估指标

#### 检索质量指标

```python
# 1. 准确率（Success Rate）
success_rate = num_correct_retrievals / total_queries

# 2. MRR（Mean Reciprocal Rank）
mrr = mean(1/rank_of_first_correct) 

# 3. NDCG@K（Normalized Discounted Cumulative Gain）
ndcg = dcg / idcg
where dcg = Σ(relevance_i / log2(i+1))

# 4. 重排改进度
improvement = (reranked_score - baseline_score) / baseline_score
```

---

## 5. 图谱 RAG（Knowledge Graph RAG）

### 5.1 知识图谱构建算法

#### 实体和关系提取

```python
# /graphrag/general/graph_extractor.py
class GraphExtractor(Extractor):
    """使用LLM进行图提取"""
    
    async def _process_single_content(self, chunk_key_dp, chunk_seq, ...):
        """
        使用LLM的few-shot提示进行实体和关系抽取
        
        LLM Prompt Structure:
        - Input text
        - Entity types (person, organization, location, etc.)
        - Output format: tuples (source, relation, target)
        - Few-shot examples
        """
        
        # 提示词结构化
        system_prompt = GRAPH_EXTRACTION_PROMPT.format(
            input_text=chunk,
            tuple_delimiter=DEFAULT_TUPLE_DELIMITER,     # "<|>"
            record_delimiter=DEFAULT_RECORD_DELIMITER,   # "##"
            entity_types=",".join(self.entity_types),
            completion_delimiter=DEFAULT_COMPLETION_DELIMITER
        )
        
        # 循环提取直到完成（ENTITY_EXTRACTION_MAX_GLEANINGS轮）
        for gleaning_round in range(self._max_gleanings):
            response = await self._llm.achat(
                system=system_prompt,
                history=[{"role": "user", "content": continuation_prompt}],
                **self._loop_args
            )
            
            # 解析响应
            entities, relationships = self._parse_response(response)
            
            # 检查是否需要继续提取
            if response.strip().upper().startswith("NO"):
                break
```

#### 关系抽取结果

```
Entity Format: entity_name<|>entity_type
Relation Format: source_entity<|>relation_type<|>target_entity<|>relation_description

Example:
"张三<|>人物"
"李四<|>人物"
"张三<|>朋友关系<|>李四<|>高中同学，已相识20年"
```

### 5.2 图遍历和路径查询

#### 实体和关系搜索

```python
class KGSearch(Dealer):
    def get_relevant_ents_by_keywords(self, keywords, filters, idxnms, kb_ids, 
                                     emb_mdl, sim_thr=0.3, N=56):
        """
        基于关键词的实体搜索
        
        Algorithm:
        1. 使用关键词生成向量
        2. 在知识图谱的实体索引中进行向量搜索
        3. 返回高相似度的实体及其描述
        """
        
        if not keywords:
            return {}
        
        filters = deepcopy(filters)
        filters["knowledge_graph_kwd"] = "entity"
        
        # 向量搜索
        matchDense = self.get_vector(", ".join(keywords), emb_mdl, 
                                    1024, sim_thr)
        
        es_res = self.dataStore.search(
            ["content_with_weight", "entity_kwd", "rank_flt"],
            [], filters, [matchDense],
            OrderByExpr(), 0, N,
            idxnms, kb_ids
        )
        
        return self._ent_info_from_(es_res, sim_thr)
    
    def get_relevant_relations_by_txt(self, txt, filters, idxnms, kb_ids, 
                                     emb_mdl, sim_thr=0.3, N=56):
        """
        基于文本的关系搜索（包括多跳关系）
        
        Returns:
        {(entity1, entity2): {
            "sim": 0.85,
            "pagerank": 0.5,
            "description": "..."
        }}
        """
```

#### 查询重写（Query Rewriting）

```python
def query_rewrite(self, llm, question, idxnms, kb_ids):
    """
    使用LLM将自然语言查询转换为实体/类型关键词
    
    Example:
    Input: "谁与张三合作过？"
    Output: 
    {
        "answer_type_keywords": ["合作", "协议"],
        "entities_from_query": ["张三"]
    }
    """
    
    # Step 1: 收集实体类型样本
    ty2ents = trio.run(lambda: get_entity_type2samples(idxnms, kb_ids))
    
    # Step 2: 构造上下文提示
    hint_prompt = PROMPTS["minirag_query2kwd"].format(
        query=question,
        TYPE_POOL=json.dumps(ty2ents, ensure_ascii=False, indent=2)
    )
    
    # Step 3: 调用LLM
    result = self._chat(llm, hint_prompt, 
                       [{"role": "user", "content": "Output:"}], {})
    
    # Step 4: 解析结果
    keywords_data = json_repair.loads(result)
    type_keywords = keywords_data.get("answer_type_keywords", [])
    entities_from_query = keywords_data.get("entities_from_query", [])[:5]
    
    return type_keywords, entities_from_query
```

### 5.3 图嵌入技术

#### Node2Vec 嵌入

```python
# /graphrag/general/entity_embedding.py
def embed_node2vec(
    graph: nx.Graph,
    dimensions: int = 1536,
    num_walks: int = 10,
    walk_length: int = 40,
    window_size: int = 2,
    iterations: int = 3,
    random_seed: int = 86,
) -> NodeEmbeddings:
    """
    使用Node2Vec生成实体嵌入
    
    Algorithm Steps:
    1. 生成随机游走（Random Walks）
    2. 对游走序列应用Skip-gram模型
    3. 学习顶点的向量表示
    
    Parameters:
    - dimensions: 嵌入维度（通常1536）
    - num_walks: 每个节点的游走数
    - walk_length: 每次游走的长度
    - window_size: Skip-gram窗口大小
    - iterations: 训练迭代次数
    """
    
    # 使用Graspologic库实现
    lcc_tensors = gc.embed.node2vec_embed(
        graph=graph,
        dimensions=dimensions,
        window_size=window_size,
        iterations=iterations,
        num_walks=num_walks,
        walk_length=walk_length,
        random_seed=random_seed,
    )
    
    return NodeEmbeddings(embeddings=lcc_tensors[0], nodes=lcc_tensors[1])
```

#### 社区检测

```python
# /graphrag/general/leiden.py
def stable_largest_connected_component(graph):
    """
    使用Leiden算法进行社区检测
    
    目的：
    1. 识别图中的社区结构
    2. 简化图的复杂性
    3. 改善嵌入质量
    """
    
    # Leiden algorithm: https://github.com/vtraag/leidenalg
    # 优于传统Louvain：更稳定，社区检测更准确
```

---

## 6. 高级特性

### 6.1 多租户隔离

```python
# 所有查询都需要指定tenant_id
def retrieval(self, question, embd_mdl, tenant_ids, kb_ids, ...):
    sres = self.search(req, 
                      [index_name(tid) for tid in tenant_ids],  # 按租户划分索引
                      kb_ids, embd_mdl, rank_feature=rank_feature)
```

### 6.2 缓存策略

```python
# LLM结果缓存（用于知识图谱查询）
def _chat(self, llm_bdl, system, history, gen_conf):
    response = get_llm_cache(llm_bdl.llm_name, system, history, gen_conf)
    if response:
        return response
    
    response = llm_bdl.chat(system, history, gen_conf)
    set_llm_cache(llm_bdl.llm_name, system, response, history, gen_conf)
    return response
```

### 6.3 跨模态检索

```python
# 支持图像和文本的混合检索
def naive_merge_with_images(texts, images, chunk_token_num=128, ...):
    """
    同时处理文本和图像chunks
    
    图像处理：
    1. 将图像转换为ID存储
    2. 在检索时动态加载
    3. 支持图像拼接
    """
```

---

## 7. 性能优化

### 7.1 批处理

```python
# 嵌入调用批处理
batch_size = 16
for i in range(0, len(texts), batch_size):
    embeddings, tokens = model.encode(texts[i:i+batch_size])
```

### 7.2 连接池

```python
# 数据库连接池
db = PooledMySQLDatabase(
    max_connections=10
)
```

### 7.3 缓存

```python
# Redis缓存
# 向量缓存、LLM响应缓存等
```

---

## 总结

RAGFlow 是一个生产级别的 RAG 框架，具有以下特点：

1. **多层混合检索**：结合全文、向量和知识图谱
2. **灵活的分块策略**：支持多种分割算法
3. **多语言支持**：内置中英文混合分词
4. **知识图谱**：完整的KG构建、嵌入和查询
5. **排序学习**：结合特征工程的结果排序
6. **生产级**：连接池、多租户、缓存等

该项目可作为构建企业级 RAG 系统的参考。
