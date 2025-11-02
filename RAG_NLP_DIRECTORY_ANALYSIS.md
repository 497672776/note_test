# RAGFlow `/rag/nlp/` 目录详解

## 📁 目录概览

```
/rag/nlp/              ← RAG 系统的 NLP 核心模块（112 KB，7 个文件）
├─ __init__.py         ← ⭐ 文档分块、分词、标记化（27 KB）
├─ search.py           ← ⭐ 混合检索引擎（27 KB）
├─ rag_tokenizer.py    ← ⭐ 分词系统（19 KB）
├─ query.py            ← ⭐ 查询处理（11 KB）
├─ term_weight.py      ← ⭐ 词权重计算（8.1 KB）
├─ synonym.py          ← 同义词处理（3.1 KB）
└─ surname.py          ← 姓氏处理（4.2 KB）
```

这是 RAGFlow **核心中的核心**，所有的文本处理都在这里！

---

## 1️⃣ `__init__.py` (27 KB) - 文档分块和分词

### 职责
```
文档处理的入口
├─ 文档分块（3 种算法）
├─ 分词和标记化
├─ 表格处理
├─ 图像处理
└─ 编号格式识别
```

### 主要函数（31 个）

#### A. 三大分块算法 ⭐⭐⭐

| 函数 | 行号 | 说明 |
|------|------|------|
| **tree_merge()** | 439-486 | 树形分块（最复杂） |
| **hierarchical_merge()** | 487-577 | 层级感知分块（推荐） |
| **naive_merge()** | 578-624 | 简单分块（最快） |
| naive_merge_with_images() | 625-721 | 支持图像的分块 |
| naive_merge_docx() | 722-800 | 支持 DOCX 的分块 |

#### B. 编号识别函数

| 函数 | 说明 |
|------|------|
| bullets_category() | 识别编号类型（第N章、1.1、#、①等） |
| qbullets_category() | PDF 中的编号识别 |
| not_bullet() | 判断是否不是编号 |
| title_frequency() | 分析编号中的标题频率 |

#### C. 分词和标记化

| 函数 | 说明 |
|------|------|
| tokenize() | 基础分词 |
| tokenize_chunks() | 对分块进行分词标记 |
| tokenize_chunks_with_images() | 带图像的分词 |
| tokenize_table() | 表格分词 |

#### D. 文本处理

| 函数 | 说明 |
|------|------|
| remove_contents_table() | 移除目录表 |
| make_colon_as_title() | 把冒号后的内容作为标题 |
| is_english() | 判断是否英文 |
| is_chinese() | 判断是否中文 |
| find_codec() | 找到合适的文字编码 |

#### E. 工具函数

| 函数 | 说明 |
|------|------|
| extract_between() | 提取标签之间的内容 |
| get_delimiters() | 解析分隔符字符串 |
| add_positions() | 添加位置信息 |
| random_choices() | 随机选择 |

#### F. 图像处理

| 函数 | 说明 |
|------|------|
| concat_img() | 拼接图像 |
| naive_merge_with_images() | 处理含图像的文档 |

#### G. 辅助类

| 类 | 说明 |
|----|------|
| Node | 树形结构节点 |

### 核心代码片段

```python
# 三大分块算法的实现位置
def tree_merge(bull, sections, depth):
    """树形分块 - 行 439"""
    # 完整树形处理

def hierarchical_merge(bull, sections, depth):
    """层级感知分块 - 行 487"""
    # 按编号规则分块

def naive_merge(sections, chunk_token_num=128, delimiter="\n。；！？", overlapped_percent=0):
    """简单分块 - 行 578"""
    # 基于 token 数的分块
```

### 数据流

```
输入：PDF/DOCX/TXT 文件
  ↓
find_codec() → 检测编码
  ↓
tokenize() → 分词
  ↓
bullets_category() → 识别编号
  ↓
naive_merge/hierarchical_merge/tree_merge() → 分块
  ↓
tokenize_chunks() → 标记每个块
  ↓
输出：chunks 列表 + tokens
```

---

## 2️⃣ `search.py` (27 KB) - 混合检索引擎

### 职责
```
实现混合检索功能
├─ 稀疏检索（全文匹配）
├─ 密集检索（向量相似度）
├─ 融合（加权组合）
├─ 重排（排序优化）
└─ 结果聚合
```

### 主要类和方法

#### Dealer 类（混合检索核心）

| 方法 | 说明 |
|------|------|
| **search()** | 执行混合检索 |
| **rerank()** | 简单重排 |
| **rerank_by_model()** | 使用 AI 模型重排 |
| **retrieval()** | 完整检索管道 |

#### Dealer 类的完整方法列表

```python
class Dealer:
    def __init__(self, data_store)
    def search()                      # 混合检索
    def rerank()                      # 重排
    def rerank_by_model()             # AI 模型重排
    def retrieval()                   # 完整检索流程
    def _rank_feature_scores()        # 排名特征计算
    def _chunk_similarity()           # 块相似度
    def hybrid_similarity()           # 混合相似度
    def get_vector()                  # 获取向量
    ...
```

### 核心算法

```python
# 混合检索的核心
def search(self, req, idx_names, kb_ids, emb_mdl=None):
    """
    1. 全文检索（稀疏）
    2. 向量检索（密集）
    3. 融合结果
    4. 重排优化
    5. 返回前 N 个结果
    """
```

### 数据流

```
查询：\"What is machine learning?\"
  ↓
分词：[\"what\", \"is\", \"machine\", \"learning\"]
  ↓
稀疏检索（全文）→ 候选列表 A
密集检索（向量）→ 候选列表 B
  ↓
融合 A + B → 候选列表 C（排序）
  ↓
重排（可选）→ 候选列表 D
  ↓
聚合（按文档） → 最终结果
  ↓
输出：Top 10 chunks + 相似度分数
```

---

## 3️⃣ `rag_tokenizer.py` (19 KB) - 分词系统

### 职责
```
文本分词（Token 级别）
├─ 中文分词（Trie 树）
├─ 英文处理（Stemming + Lemmatization）
├─ 词性标注
├─ 细粒度分词
└─ 自定义词表
```

### 主要类和函数

#### RagTokenizer 类（核心分词器）

| 方法 | 说明 |
|------|------|
| **tokenize()** | 基础分词 |
| **fine_grained_tokenize()** | 细粒度分词 |
| **tag()** | 词性标注 |
| **split_by_lang()** | 中英混合分割 |

#### 全局函数

| 函数 | 说明 |
|------|------|
| is_chinese() | 判断中文字符 |
| is_number() | 判断数字 |
| is_alphabet() | 判断字母 |
| naiveQie() | 简单分词 |

### 核心算法

```python
# 中文分词的三步法
def tokenize(self, line):
    """
    Step 1: 预处理（全角→半角，繁→简）
    Step 2: 语言识别（中/英分割）
    Step 3: 中文分词（Trie 树 + DFS）
    Step 4: 英文处理（Stemming + Lemmatization）
    Step 5: 细粒度分词
    """
```

### 数据流

```
输入文本：\"自然语言处理技术\"
  ↓
预处理：全角→半角，繁→简
  ↓
Trie 树查找：自然|语言|处理|技术
  ↓
细粒度分词：进一步拆分复合词
  ↓
词性标注：[('自然', NN), ('语言', NN), ...]
  ↓
输出 tokens：[\"自然\", \"语言\", \"处理\", \"技术\"]
```

---

## 4️⃣ `query.py` (11 KB) - 查询处理

### 职责
```
处理用户查询
├─ 分词查询
├─ 全文查询
├─ 关键词提取
├─ 查询重写
└─ 同义词扩展
```

### 主要类

#### FulltextQueryer 类

| 方法 | 说明 |
|------|------|
| **question()** | 处理问题查询 |
| **paragraph()** | 处理段落查询 |
| **keywords()** | 提取关键词 |
| **rewrite()** | 查询重写 |

### 核心功能

```python
class FulltextQueryer:
    def __init__(self):
        self.tw = term_weight.Dealer()      # 词权重
        self.syn = synonym.Dealer()         # 同义词
        self.query_fields = [\"title\", \"content\", \"keywords\"]

    def question(self, qst, min_match=0.3):
        \"\"\"
        处理自然语言问题
        - 分词
        - 词权重计算
        - 同义词扩展
        \"\"\"

    def paragraph(self, txt, important_keywords=[]):
        \"\"\"处理段落查询\"\"\"

    def keywords(self, txt, topn=10):
        \"\"\"提取关键词\"\"\"
```

### 数据流

```
查询：\"什么是深度学习？\"
  ↓
分词：[\"什么\", \"是\", \"深度\", \"学习\"]
  ↓
词权重：[(\"深度\", 0.4), (\"学习\", 0.6)]
  ↓
同义词扩展：添加 \"neural network\", \"deep learning\" 等
  ↓
生成全文查询表达式
  ↓
输出：MatchTextExpr（发送到搜索引擎）
```

---

## 5️⃣ `term_weight.py` (8.1 KB) - 词权重计算

### 职责
```
计算每个词的权重
├─ IDF（逆文档频率）
├─ TF（词频）
├─ NER（实体识别）
├─ POS（词性标注）
└─ 综合权重
```

### 主要类

#### Dealer 类

| 方法 | 说明 |
|------|------|
| **weights()** | 计算词权重 |
| **idf()** | 计算 IDF |
| **tf()** | 计算 TF |

### 权重公式

```python
def weights(self, tks):
    \"\"\"
    权重 = (0.3*IDF_freq + 0.7*IDF_df) × NER_factor × POS_factor

    IDF_freq = log(总文档数 / 包含词的文档数)
    IDF_df = TF × IDF
    NER_factor:
        - 公司/地点：×3
        - 数值：×2
        - 其他：×1
    POS_factor:
        - 名词：×2
        - 地点/机构：×3
        - 代词：×0.3
    \"\"\"
```

### 数据流

```
输入词语：[\"人工智能\", \"深度\", \"学习\"]
  ↓
计算 TF（词频）
  ↓
计算 IDF（逆文档频率）
  ↓
NER 识别：\"人工智能\" → 技术名词 → ×1.5
  ↓
POS 标注：\"深度\" → 名词 → ×2
  ↓
综合权重：[(\"人工智能\", 0.45), (\"深度\", 0.35), ...]
  ↓
归一化：所有权重之和 = 1.0
  ↓
输出：词权重列表
```

---

## 6️⃣ `synonym.py` (3.1 KB) - 同义词处理

### 职责
```
处理同义词和词语扩展
├─ 同义词字典
├─ 词语替换
└─ 查询扩展
```

### 主要类

#### Dealer 类

| 方法 | 说明 |
|------|------|
| **get_synonyms()** | 获取同义词 |
| **expand()** | 扩展查询词 |

### 示例

```python
# 示例
synonyms = {
    \"深度学习\": [\"neural network\", \"DL\"],
    \"人工智能\": [\"AI\", \"machine intelligence\"],
    \"自然语言处理\": [\"NLP\", \"text processing\"]
}

# 查询 \"深度学习\"
# 扩展为 [\"深度学习\", \"neural network\", \"DL\"]
# 提高检索覆盖率
```

---

## 7️⃣ `surname.py` (4.2 KB) - 姓氏处理

### 职责
```
处理人名和姓氏
├─ 姓氏识别
├─ 人名分割
└─ 中文名字处理
```

### 用处
```
分词时，识别 \"张三\" 应该保留，而不是拆成 \"张\" 和 \"三\"
```

---

## 📊 模块之间的关系

```
输入文本
  ↓
__init__.py (分块和分词)
  ├─ tokenize() → rag_tokenizer.py (分词)
  ├─ bullets_category() → 识别编号
  └─ naive/hierarchical/tree_merge() → 分块
  ↓
搜索查询
  ↓
query.py (查询处理)
  ├─ FulltextQueryer.question() → 分词
  ├─ term_weight.py (词权重)
  └─ synonym.py (同义词扩展)
  ↓
search.py (混合检索)
  ├─ 稀疏检索（全文）
  ├─ 密集检索（向量）
  ├─ 融合
  └─ 重排
  ↓
输出结果
```

---

## 🎯 核心数据流全景

```
┌─────────────────────────────────────────────────────────────────────┐
│                          文档处理流程                               │
└─────────────────────────────────────────────────────────────────────┘

输入：PDF/DOCX/TXT
  ↓
[__init__.py] find_codec() → 检测编码
  ↓
[__init__.py] tokenize() + [rag_tokenizer.py] RagTokenizer → 分词
  ↓
[__init__.py] bullets_category() → 识别编号格式
  ↓
[__init__.py] naive/hierarchical/tree_merge() → 文档分块
  ↓
[__init__.py] tokenize_chunks() → 对每个块分词标记
  ↓
输出：chunks + tokens

┌─────────────────────────────────────────────────────────────────────┐
│                          查询和检索流程                             │
└─────────────────────────────────────────────────────────────────────┘

输入查询：\"What is machine learning?\"
  ↓
[query.py] FulltextQueryer.question() → 处理查询
  ├─ [rag_tokenizer.py] 分词
  ├─ [term_weight.py] 计算词权重
  └─ [synonym.py] 扩展同义词
  ↓
[search.py] Dealer.search() → 混合检索
  ├─ 稀疏检索：全文匹配
  ├─ 密集检索：向量相似度
  ├─ 融合：加权组合
  └─ 重排：AI 模型优化
  ↓
输出：Top N 相关 chunks + 分数
```

---

## 💡 关键数据结构

### 在 `__init__.py` 中

```python
# chunks 的结构
chunks = [
    {
        'content': '文本内容',
        'tokens': ['分词', '结果'],
        'token_count': 512,
        'bullet': 0,  # 编号类型
        'level': 2    # 层级
    },
    ...
]

# Node 类（树形结构）
class Node:
    def __init__(self, text, depth):
        self.text = text
        self.depth = depth
        self.children = []
```

### 在 `search.py` 中

```python
# 搜索结果的结构
@dataclass
class SearchResult:
    ids: list[str]          # chunk IDs
    field: dict             # chunk 内容
    keywords: list[str]     # 关键词
    aggregation: dict       # 按文档聚合统计
    scores: list[float]     # 相似度分数
```

---

## 🔗 文件大小和复杂度

| 文件 | 大小 | 函数/类 数 | 复杂度 | 关键程度 |
|------|------|-----------|--------|--------|
| __init__.py | 27 KB | 31 func + 1 class | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| search.py | 27 KB | 1 class (20+ 方法) | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| rag_tokenizer.py | 19 KB | 1 class + 4 func | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| query.py | 11 KB | 1 class (5+ 方法) | ⭐⭐ | ⭐⭐⭐ |
| term_weight.py | 8.1 KB | 1 class (3+ 方法) | ⭐⭐ | ⭐⭐⭐ |
| synonym.py | 3.1 KB | 1 class (2 方法) | ⭐ | ⭐⭐ |
| surname.py | 4.2 KB | 1 class | ⭐ | ⭐ |

---

## 📚 学习路线

### 初级（理解概念）
```
1. 了解三大分块算法 → __init__.py (439-624 行)
2. 理解分词系统 → rag_tokenizer.py
3. 学习词权重 → term_weight.py
```

### 中级（深入算法）
```
1. 完整读 __init__.py
2. 学习混合检索 → search.py
3. 学习查询处理 → query.py
```

### 高级（优化和扩展）
```
1. 优化三大算法的性能
2. 添加新的编号格式支持
3. 集成新的同义词库
4. 实现自定义权重函数
```

---

## 🎯 快速导航

需要找什么，看这里：

| 需求 | 文件 | 行号 |
|------|------|------|
| 简单分块代码 | __init__.py | 578-624 |
| 层级分块代码 | __init__.py | 487-577 |
| 树形分块代码 | __init__.py | 439-486 |
| 分词实现 | rag_tokenizer.py | 全部 |
| 混合检索实现 | search.py | 全部 |
| 词权重公式 | term_weight.py | 全部 |
| 查询处理 | query.py | 全部 |

---

**这就是 RAGFlow 的 NLP 核心！** 🚀

所有的文本处理、分块、分词、检索都在这 7 个文件里。
