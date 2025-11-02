# RAGFlow 详细工作流程指南

## 目录
1. [文档上传工作流](#文档上传工作流)
   - [文档解析详解（manual.py）](#第-25-步文档解析manualpy-实现详解)
   - [PDF 处理流程](#pdf-解析流程pdf-类)
   - [DOCX 处理流程](#docx-解析流程docx-类)
2. [对话查询工作流](#对话查询工作流)
3. [系统架构数据流](#系统架构数据流)
4. [性能分析](#性能分析)

---

## 文档上传工作流

### 简单流程图

```
用户上传 PDF
   ↓
PostgreSQL: INSERT documents (状态=0)
   ↓
MinIO: 存储原始文件
   ↓
PostgreSQL: UPDATE documents (状态=1，处理中)
   ↓
后台任务: 解析 PDF → 生成 chunks
   ↓
Elasticsearch: 批量插入 chunks + vectors
   ↓
PostgreSQL: UPDATE documents (状态=2，完成)
```

### 详细步骤分解

用户上传 **product_manual.pdf**（50MB）

#### 第 1 步：保存原始文件

```
文件：product_manual.pdf (50MB)
存储位置：MinIO /documents/original/
原因：
  - 太大了，不能存数据库
  - 需要长期保存（用户随时可能下载）

[MinIO 写入：5-10 秒]
```

**为什么不存 PostgreSQL？**
- PostgreSQL 是关系数据库，设计用于结构化数据
- 存储大二进制文件会拖累数据库性能
- 数据库连接会被占用，影响并发

**为什么选择 MinIO？**
- S3 兼容的对象存储
- 支持大文件（无大小限制）
- 支持版本控制和生命周期管理
- 独立扩展，不影响其他组件

---

#### 第 2 步：OCR 识别（如果是扫描版 PDF）

```
过程：逐页识别文字
每页结果存在：MinIO /cache/ocr_results/
             /page_001.json
             /page_002.json
             ...

[OCR 处理：2-3 分钟，生成 100MB+ 结果]
```

**什么时候用 OCR？**

OCR（Optical Character Recognition，光学字符识别）用于将图像中的文字识别出来。

```
PDF 分为两大类：
│
├─ 文本 PDF（可复制）
│   ├ 特点：PDF 已包含文本信息
│   ├ 大小：通常 1-5MB
│   ├ 处理：直接提取文字，无需 OCR
│   └ 用时：< 1 秒
│
└─ 扫描 PDF（图像）
    ├ 特点：实际是 JPEG 叠层
    ├ 大小：通常 20-100MB（高分辨率）
    ├ 处理：需要 OCR 识别文字
    ├ 用时：2-3 分钟（取决于页数和分辨率）
    │
    └ 例子：
        - 扫描的合同
        - 扫描的报告
        - 手写笔记扫描件
        - 旧书籍的数字化
```

**OCR 在 RAGFlow 中的流程**：

```
1. 检测：识别这是扫描 PDF
   └─ 方法：尝试提取文本，如果为空则是扫描版

2. 预处理：增强图像质量
   ├─ 倾斜矫正（Deskew）
   ├─ 去噪（Denoise）
   └─ 二值化（Binarization）

3. OCR 识别：逐页识别
   ├─ 使用：Tesseract 或 PaddleOCR
   ├─ 输出：每页提取的文本 + 置信度 + 位置信息
   └─ 格式：
       {
           "page": 1,
           "text": "产品说明书\n第一章：基本介绍",
           "confidence": 0.92,
           "blocks": [
               {
                   "text": "产品说明书",
                   "bbox": [100, 50, 400, 100],
                   "confidence": 0.95
               }
           ]
       }

4. 结果保存：中间文件
   └─ 位置：MinIO /cache/ocr_results/page_*.json
   └─ 原因：临时中间文件，处理完后可删除
   └─ 保留时间：30 天（自动过期）
```

**OCR 结果的后续使用**：

```
OCR 识别的文本
         ↓
┌─────────────────────────────────┐
│ 与原 PDF 文本合并                │
│（如果 PDF 既有扫描又有文本）      │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│ 分词、分句                        │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│ 按 chunk_method 切割              │
│（naive_merge / hierarchical）    │
└─────────────────────────────────┘
         ↓
┌─────────────────────────────────┐
│ 生成向量并存入 Elasticsearch      │
└─────────────────────────────────┘
```

---

#### 第 2.5 步：文档解析（manual.py 实现详解）

RAGFlow 的文档解析核心逻辑在 `ragflow/rag/app/manual.py`，支持 PDF 和 DOCX 两种格式。

**PDF 解析流程（Pdf 类）**：

```python
class Pdf(PdfParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, zoomin=3, callback=None):

        # 第 1 阶段：OCR 识别（如果是扫描 PDF）
        self.__images__(
            filename if not binary else binary,
            zoomin,
            from_page,
            to_page,
            callback
        )
        callback(msg="OCR finished")

        # 第 2 阶段：版面分析（识别标题、正文、表格位置）
        self._layouts_rec(zoomin)
        callback(0.65, "Layout analysis")

        # 第 3 阶段：表格识别和提取
        self._table_transformer_job(zoomin)
        callback(0.67, "Table analysis")

        # 第 4 阶段：文本合并和清理
        self._text_merge()
        tbls = self._extract_table_figure(True, zoomin, True, True)
        self._concat_downward()
        self._filter_forpages()
        callback(0.68, "Text merged")

        # 清理多余空格
        for b in self.boxes:
            b["text"] = re.sub(r"([\t 　]|\u3000){2,}", " ", b["text"].strip())

        # 返回：[(文本, 版面号, 位置信息), ...], 表格列表
        return [(b["text"], b.get("layoutno", ""), self.get_position(b, zoomin))
                for i, b in enumerate(self.boxes)], tbls
```

**PDF 处理流程的 4 个阶段**：

```
┌──────────────────────────────────────────────────────────┐
│ 阶段 1：OCR 识别（Optical Character Recognition）       │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ 用途：将 PDF 中的图像转换为可搜索的文本                  │
│                                                          │
│ 工作原理：                                               │
│ 1. 按设定的缩放倍数（zoomin）放大页面                   │
│    - zoomin=3 表示 300% 放大，更清晰的 OCR              │
│    - 放大后对低分辨率 PDF 有帮助                        │
│                                                          │
│ 2. 逐页调用 OCR 引擎（Tesseract/PaddleOCR）            │
│    - 识别文字并标记位置（bounding box）                 │
│    - 识别文字方向（左->右、上->下、竖直等）            │
│                                                          │
│ 3. 生成 boxes 列表                                       │
│    boxes = [                                             │
│        {                                                 │
│            "text": "第一章 基本介绍",                  │
│            "x0": 100,                                    │
│            "y0": 50,                                     │
│            "x1": 400,                                    │
│            "y1": 100,                                    │
│            "page": 0                                     │
│        },                                                │
│        ...                                               │
│    ]                                                     │
│                                                          │
│ 耗时：大多数时间花在这里（2-3 分钟）                    │
└──────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段 2：版面分析（Layout Recognition）                  │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ 用途：识别 PDF 中的结构（标题、正文、表格、图片）       │
│                                                          │
│ 工作原理：                                               │
│ 1. 使用深度学习模型识别区域类型                         │
│    - Text（正文）                                       │
│    - Title（标题）                                      │
│    - Table（表格）                                      │
│    - Figure（图片）                                     │
│    - Header/Footer（页眉页脚）                          │
│                                                          │
│ 2. 按逻辑顺序排列 boxes                                  │
│    - 从上到下，从左到右                                 │
│    - 标题优先级最高                                     │
│                                                          │
│ 3. 记录版面信息（layoutno）                             │
│    - 用于后续的层级识别                                 │
│                                                          │
│ 例子：                                                   │
│ Box 1: "第一章"    → layoutno = "title_level_1"         │
│ Box 2: "第 1.1 节" → layoutno = "title_level_2"         │
│ Box 3: "本节内容..." → layoutno = "text"                │
│ Box 4: "[表格]"    → layoutno = "table"                 │
│                                                          │
│ 耗时：50-100ms（相对快速）                              │
└──────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段 3：表格识别（Table Analysis）                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ 用途：识别和提取表格结构                                 │
│                                                          │
│ 工作原理：                                               │
│ 1. 检测表格边界（线条识别）                             │
│    - 水平线、竖直线的交点形成单元格                     │
│                                                          │
│ 2. 识别单元格合并（colspan, rowspan）                  │
│    - 检测相同的单元格内容 → 标记为合并                 │
│                                                          │
│ 3. 提取表格数据                                         │
│    - 每行 → [单元格1, 单元格2, ...]                     │
│    - 返回 HTML 格式：<table><tr><td>...</td></tr></table>│
│                                                          │
│ 输出格式：                                               │
│ tbls = [                                                │
│     ((img, html_str), page_info),                       │
│     ...                                                  │
│ ]                                                        │
│ 其中：                                                   │
│   - img: 表格的图片（用于显示）                         │
│   - html_str: 表格 HTML（用于文本提取）                │
│   - page_info: 页码和位置信息                           │
│                                                          │
│ 耗时：100-200ms                                         │
└──────────────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────────────┐
│ 阶段 4：文本合并和清理（Text Merge）                    │
├──────────────────────────────────────────────────────────┤
│                                                          │
│ 用途：将零散的 boxes 组合成完整的文本段落              │
│                                                          │
│ 工作原理：                                               │
│ 1. _text_merge()：按逻辑顺序合并 boxes                 │
│    - 同一段落的 boxes 合并成一行                        │
│    - 保留段落之间的换行符                               │
│                                                          │
│ 2. _extract_table_figure()：提取表格和图片             │
│    - 分离出表格和图片数据                               │
│    - 减少干扰信息                                       │
│                                                          │
│ 3. _concat_downward()：垂直合并                        │
│    - 将相邻的 boxes 合并                                │
│    - 例如：["标题1", "内容"] → ["标题1\n内容"]          │
│                                                          │
│ 4. 清理空格                                             │
│    - 去除多余的空格、制表符、全角空格                   │
│    - re.sub(r"([\t 　]|\u3000){2,}", " ", text)        │
│                                                          │
│ 输出：                                                   │
│ sections = [                                            │
│     ("第一章 基本介绍\n本章介绍...", "title_level_1",   │
│      [(0, 100, 50, 400, 100), ...]),  # 位置信息         │
│     ...                                                  │
│ ]                                                        │
│                                                          │
│ 耗时：50-100ms                                          │
└──────────────────────────────────────────────────────────┘
```

**DOCX 解析流程（Docx 类）**：

```python
class Docx(DocxParser):
    def __call__(self, filename, binary=None, from_page=0,
                 to_page=100000, callback=None):

        # 加载 DOCX 文件
        self.doc = Document(filename) if not binary else Document(BytesIO(binary))

        ti_list = []  # (text, image) 对
        question_stack = []  # 当前的问题堆栈（多级标题）
        level_stack = []  # 当前的级别堆栈

        for p in self.doc.paragraphs:
            # 检查段落是否是标题（通过 docx_question_level 函数）
            question_level, p_text = docx_question_level(p)

            if not question_level or question_level > 6:  # 不是标题，是内容
                # 累积答案文本
                last_answer = f'{last_answer}\n{p_text}'
                # 累积该段的图片
                current_image = self.get_picture(self.doc, p)
                last_image = self.concat_img(last_image, current_image)
            else:  # 是标题
                # 保存之前的答案
                if last_answer or last_image:
                    sum_question = '\n'.join(question_stack)
                    if sum_question:
                        ti_list.append((f'{sum_question}\n{last_answer}', last_image))

                # 维护问题堆栈（去除比当前级别低的问题）
                while question_stack and question_level <= level_stack[-1]:
                    question_stack.pop()
                    level_stack.pop()

                # 添加当前标题
                question_stack.append(p_text)
                level_stack.append(question_level)

        # 处理最后一个段落
        if last_answer:
            sum_question = '\n'.join(question_stack)
            if sum_question:
                ti_list.append((f'{sum_question}\n{last_answer}', last_image))

        # 提取表格
        tbls = []
        for tb in self.doc.tables:
            html = "<table>"
            for r in tb.rows:
                html += "<tr>"
                for c in r.cells:
                    html += f"<td>{c.text}</td>"
                html += "</tr>"
            html += "</table>"
            tbls.append(((None, html), ""))

        return ti_list, tbls
```

**DOCX 处理流程的关键点**：

```
DOCX 特点：结构化格式，包含段落、表格、图片等元素

解析思路：
├─ 问题堆栈方式识别层级
│  ├─ 读取 <w:pStyle val="Heading1"/> 等样式
│  ├─ 问题级别 (question_level) 决定了层级深度
│  │  - level=1: 最高级标题（章）
│  │  - level=2: 次级标题（节）
│  │  - level=3: 更低级（小节）
│  │  - level>6: 不是标题，是正文
│  │
│  └─ 堆栈维护：保存当前的问题链路
│     问题堆栈示例：
│     ["第一章", "第 1.1 节", "第 1.1.1 小节"]
│                 ↑
│         当前正在读这些内容的答案
│
├─ 答案累积
│  ├─ 读到新的标题时，保存之前的答案
│  ├─ 同时合并堆栈中的所有问题（多级标题）
│  └─ 输出格式：
│     "第一章\n第 1.1 节\n第 1.1.1 小节\n[内容文本]"
│
└─ 图片处理
   ├─ 每段可能包含图片
   ├─ 使用 get_picture() 提取
   ├─ 使用 concat_img() 合并多张图片
   └─ 与文本一起保存为 (text, image) 对

例子：
原始 DOCX：
  # 第一章：基础概念
  这一章讲解基础知识
  [图片1]
  ## 第 1.1 节：定义
  定义就是...
  [图片2]
  内容继续...

处理结果：
  ti_list = [
      ("第一章：基础概念\n这一章讲解基础知识", 图片1),
      ("第一章：基础概念\n第 1.1 节：定义\n定义就是...", 图片2),
      ("第一章：基础概念\n第 1.1 节：定义\n内容继续...", None)
  ]
```

**chunk() 函数的分块逻辑**：

```python
def chunk(filename, binary=None, from_page=0, to_page=100000,
          lang="Chinese", callback=None, **kwargs):

    parser_config = kwargs.get("parser_config", {
        "chunk_token_num": 512,           # 每个 chunk 目标 token 数
        "delimiter": "\n!?。；！？",      # 分句符号
        "layout_recognize": "DeepDOC"     # 版面识别方法
    })

    # 处理文件名（提取文档名）
    doc["docnm_kwd"] = filename
    doc["title_tks"] = rag_tokenizer.tokenize(filename)

    if re.search(r"\.pdf$", filename, re.IGNORECASE):
        # PDF 解析...
        sections, tbls = pdf_parser(filename, ...)

        # 第 1 步：识别章节级别（section_id）
        # 目的：区分不同的主要段落（通常按标题分）
        sec_ids = []
        sid = 0
        for i, lvl in enumerate(levels):
            if lvl <= most_level and i > 0 and lvl != levels[i - 1]:
                sid += 1  # 遇到新的主标题，section_id 增加
            sec_ids.append(sid)

        # 第 2 步：合并分块（chunk merging）
        chunks = []
        last_sid = -2
        tk_cnt = 0
        for txt, sec_id, poss in sorted(sections, ...):
            # 合并逻辑：
            # - 如果 token 数 < 32，强制合并（太小的内容）
            # - 如果 token 数 < 1024 且在同一 section，继续合并
            # - 否则创建新 chunk

            if tk_cnt < 32 or (tk_cnt < 1024 and (sec_id == last_sid or sec_id == -1)):
                chunks[-1] += "\n" + txt + poss  # 合并到上一个 chunk
                tk_cnt += num_tokens_from_string(txt)
            else:
                chunks.append(txt + poss)  # 创建新 chunk
                tk_cnt = num_tokens_from_string(txt)
                last_sid = sec_id

        # 第 3 步：分词和向量化
        res = tokenize_table(tbls, doc, eng)
        res.extend(tokenize_chunks(chunks, doc, eng, pdf_parser))
        return res

    elif re.search(r"\.docx?$", filename, re.IGNORECASE):
        # DOCX 解析...
        ti_list, tbls = docx_parser(filename, ...)

        # 处理每个 (text, image) 对
        for text, image in ti_list:
            d = copy.deepcopy(doc)
            if image:
                d['image'] = image
                d["doc_type_kwd"] = "image"
            tokenize(d, text, eng)
            res.append(d)
        return res
```

**分块策略详解**：

```
目标：平衡内容完整性和 chunk 大小

token_num = 512  （默认配置）

分块逻辑流程：

第 1 条内容（标题）
  200 tokens  ┐
              ├─ 总 300 tokens < 512 → 继续合并
第 2 条内容  │
  100 tokens ┘
              → 合并到一个 chunk

第 3 条内容
  600 tokens  → 超过 512，创建新 chunk

结果：
  Chunk 1: (200 + 100 = 300 tokens)
  Chunk 2: (600 tokens)

边界情况处理：
├─ token < 32：强制合并（太小的内容，保留上下文）
├─ 32 ≤ token < 1024：
│  ├─ 如果与上一个在同一 section → 继续合并
│  └─ 如果是表格（sec_id == -1）→ 继续合并
├─ token ≥ 1024：创建新 chunk（即使有内容要合并）
└─ 目的：避免 chunk 过大导致嵌入模型溢出
```

---

#### 第 3 步：分词和向量生成

```
处理：将 PDF 切成 1000 个 chunks
生成：1000 个 768 维向量
每个向量大小：768 × 4 字节 = 3KB
总大小：1000 × 3KB = 3MB

存在：MinIO /embeddings/doc_product_manual_*.bin

[向量生成：1-2 分钟，3MB]
```

**数据流**：

```
OCR 结果（或 PDF 文本）
         ↓
分词（使用 jieba、spaCy 等）
例如："产品说明书很详细" → ["产品", "说明书", "很", "详细"]
         ↓
分句（按句号、分号分割）
         ↓
分 chunks（按 chunk_method）
例如：
  - naive_merge：简单合并直到达到 token_count 限制
  - hierarchical_merge：保持段落结构
  - tree_merge：保持树形层级（用于法律文档）
         ↓
生成向量
使用 BAAI/bge-large-zh 等模型
每个 chunk → 768 维向量
         ↓
保存向量
位置：MinIO /embeddings/
格式：二进制 .bin 文件（节省空间）
```

---

#### 第 4 步：存入 Elasticsearch

```
数据：1000 个 chunks（包括向量、文本、元数据）
存入：Elasticsearch chunks index

[批量索引：30-60 秒]
```

**Elasticsearch 中的索引结构**：

```
chunk_index
│
├─ chunk_001
│  ├─ chunk_id: "chunk_abc123"
│  ├─ doc_id: "doc_product_manual"
│  ├─ kb_id: "kb_456"
│  ├─ content: "产品说明书的第一章..."
│  ├─ vector: [0.123, -0.456, ..., 0.789]  ← 768 维向量
│  ├─ page_num: 1
│  ├─ position_in_doc: 0
│  └─ created_time: "2025-11-02T15:30:00Z"
│
├─ chunk_002
│  └─ ...
│
└─ chunk_nnn
   └─ ...
```

---

#### 第 5 步：完成，MinIO 最终保留什么？

```
✓ /documents/original/product_manual.pdf
  （永久保存，用户资产）

✗ /cache/ocr_results/...
  （自动删除，临时中间文件）

✗ /embeddings/doc_product_manual_*.bin
  （可以删除，可以从 ES 重新生成）
```

---

## 对话查询工作流

### 简单流程图

```
用户问题："什么是 RAG？"
   ↓
[Redis] 检查缓存（2ms）
   ├─ 缓存命中 → 直接返回（2ms）
   └─ 缓存未命中 → 继续
      ↓
      [本地] 生成查询向量（10-50ms）
         ↓
      [Elasticsearch] 向量搜索（50-200ms）
         ↓
      [可选] 重排（如果启用）（100-500ms）
         ↓
      [Redis] 缓存结果（2ms）
         ↓
      返回给用户
```

### 详细步骤分解

#### 第 1 步：生成缓存键

```python
question = "什么是 RAG？"
kb_ids = ["kb_123", "kb_456"]
top_k = 10

# 生成唯一的缓存键
cache_key = md5("什么是 RAG？_kb_123,kb_456_10".encode())
          = "6a9f8e2c8d4b1a5f..."（32 字符 hash）

[生成缓存键用时：<1ms]
```

**为什么要生成缓存键？**
- Redis 中不能直接用长文本作为 key（效率低）
- 使用 MD5 hash 保证唯一性且长度固定
- 相同问题 + 相同知识库 = 相同 cache_key

---

#### 第 2 步：检查 Redis 缓存

```
命令：redis.get("search:6a9f8e2c8d4b1a5f...")

[Redis 查询用时：2ms]
```

**缓存键在 Redis 中存的数据**：

```json
{
    "results": [
        {
            "chunk_id": "chunk_123",
            "content": "RAG 是检索增强生成，通过检索...",
            "doc_id": "doc_001",
            "similarity": 0.92,
            "page_num": 5,
            "metadata": {
                "source": "introduction.pdf"
            }
        },
        // ... 最多 10 条
    ],
    "total": 10,
    "search_time_ms": 87,
    "cached_at": "2025-11-02 15:30:45"
}
```

**缓存命中率分析**：

```
场景 A：热门问题（30-50% 缓存命中率）
├─ 例子："什么是 RAG？"、"如何使用 API？"
├─ 原因：相同问题被频繁提问
├─ 结果：2ms 返回
└─ 影响：系统响应快，用户体验好

场景 B：冷门问题（50-70% 缓存未命中）
├─ 例子："根据文件内容生成自定义报告"
├─ 原因：问法各异，缓存难以复用
├─ 结果：需要完整搜索流程，100-300ms
└─ 影响：首次查询慢，之后快

缓存策略影响：
├─ 缓存时间短（5分钟）
│  └─ 优点：数据新鲜度高，不会输出陈旧信息
│  └─ 缺点：缓存命中率低，热门问题也要重新搜索
│
└─ 缓存时间长（24小时）
   └─ 优点：缓存命中率高，系统压力小
   └─ 缺点：文档更新后用户仍看到旧回答
```

---

#### 第 3 步：生成查询向量（缓存未命中时）

```
模型加载（第一次会加载到内存，之后复用）：
model = load_model("BAAI/bge-large-zh@Builtin")

生成向量：
query_vector = model.encode("什么是 RAG？")
             = [0.123, -0.456, 0.789, ..., 0.234]
             = 768 个浮点数
             = 大小：768 × 4 字节 = 3KB

[向量生成用时：10-50ms（取决于硬件）]
```

**向量模型的选择**：

```
不同模型对比：

BAAI/bge-large-zh（中文大模型）
├─ 维度：1024
├─ 适用：中文文档、中文问题
├─ 性能：0.95 M-NDCG@10（业界最好）
└─ 用时：20-40ms

BAAI/bge-small-en-v1.5（英文小模型）
├─ 维度：384
├─ 适用：英文文档、快速回应
├─ 性能：0.85 M-NDCG@10
└─ 用时：5-10ms

OpenAI text-embedding-3-large
├─ 维度：3072
├─ 适用：通用、高精度
├─ 性能：0.92 M-NDCG@10
├─ 用时：500-1000ms（需要调用 API）
└─ 成本：$0.02 / 1M tokens
```

---

#### 第 4 步：向量数据库搜索（Elasticsearch）

```
调用：results = vector_db.search(
          question="什么是 RAG？",
          query_vector=[0.123, ..., 0.234],
          kb_ids=["kb_123", "kb_456"],
          top_k=10,
          weights=(0.5, 0.5)  # 向量50% + 关键词50%
      )

[向量数据库搜索用时：50-200ms]
```

**Elasticsearch 内部搜索流程**：

```
Step A：向量搜索（KNN - K-Nearest Neighbors）
├─ 原理：找距离最近的 chunks
├─ 算法：余弦相似度计算
├─ 公式：similarity = (A·B) / (|A| × |B|)
│         其中 A = 查询向量，B = chunk 向量
├─ 结果例子：
│   ├─ chunk_1: 相似度 0.92 ✓ 相关
│   ├─ chunk_2: 相似度 0.87 ✓ 相关
│   ├─ chunk_3: 相似度 0.45 ✗ 不相关
│   └─ chunk_4: 相似度 0.38 ✗ 不相关
│
└─ 返回：Top-100 chunks（未排序）

Step B：关键词搜索（BM25 算法）
├─ 过程：
│   1. 分词："什么"、"是"、"RAG"
│   2. 查找包含这些词的 chunks
│   3. 计算 BM25 分数（考虑词频和文档频率）
│
├─ BM25 公式概念：
│   分数 = Σ IDF(词) × (词频 × (k1+1)) / (词频 + k1(1-b+b×doc_length/avgdoc_length))
│
├─ 结果例子：
│   ├─ chunk_1: BM25 分数 0.85 ✓
│   ├─ chunk_2: BM25 分数 0.72 ✓
│   ├─ chunk_3: BM25 分数 0.35 ✗
│   └─ chunk_4: BM25 分数 0.28 ✗
│
└─ 返回：Top-100 chunks（未排序）

Step C：融合分数（Hybrid Search）
├─ 公式：
│   final_score = vector_weight × vector_score
│               + keyword_weight × keyword_score
│
├─ 例子（weights = 0.5, 0.5）：
│   chunk_1: 0.92 × 0.5 + 0.85 × 0.5 = 0.885 ✓
│   chunk_2: 0.87 × 0.5 + 0.72 × 0.5 = 0.795 ✓
│   chunk_3: 0.45 × 0.5 + 0.35 × 0.5 = 0.400 ~
│
└─ 最后按 final_score 降序排列，取 Top-10

返回给用户：
{
    "chunks": [
        {
            "chunk_id": "chunk_123",
            "content": "RAG 是检索增强生成...",
            "doc_id": "doc_456",
            "vector_score": 0.92,
            "keyword_score": 0.85,
            "final_score": 0.885,
            "metadata": {...}
        },
        // ... 9 条更多
    ],
    "total": 10,
    "search_time_ms": 87
}
```

**向量权重 vs 关键词权重**：

```
场景 1：精准语义搜索（权重 0.8:0.2）
├─ 使用：回答概念性问题
├─ 例子："什么是 RAG？"
├─ 效果：重视语义相似，忽略关键词完全匹配
└─ 原因：同一概念可用不同词语表达

场景 2：精准关键词匹配（权重 0.2:0.8）
├─ 使用：搜索具体数据或术语
├─ 例子："产品编号是多少？"
├─ 效果：优先返回包含"产品编号"的 chunks
└─ 原因：用户明确知道要搜什么词

场景 3：平衡搜索（权重 0.5:0.5）← 推荐
├─ 使用：大多数通用场景
├─ 优点：既能理解语义，又能精准定位
└─ 缺点：两者都不是最优
```

---

#### 第 5 步：可选的重排（Reranking）

```
如果配置了 reranker_model（如 Cohere、Jina）：

reranker.rerank(
    query="什么是 RAG？",
    documents=[chunk.content for chunk in chunks],
    top_k=10
)

→ 返回重新排序的 chunks（通常更精准）

[重排用时：100-500ms（如果用 API 型 reranker）]
```

**重排器的作用**：

```
问题：为什么还需要重排？

原因：Elasticsearch 的评分不够精准
├─ 它是根据"相似度"和"词频"计算
├─ 但这两个因素并不一定代表"对问题的回答质量"
│
└─ 例子：
   问题："RAG 系统如何处理长文档？"

   ES 返回的排名：
   1. chunk_1: "RAG 是一种技术..." ← 包含 RAG，但不相关
   2. chunk_2: "处理长文档需要..." ← 包含关键词，但不是 RAG 特定
   3. chunk_3: "RAG 通过分块处理长文档..." ← 最相关，但排名最低

   Reranker 重新排名后：
   1. chunk_3: "RAG 通过分块处理长文档..." ← 最相关
   2. chunk_2: "处理长文档需要..." ← 次相关
   3. chunk_1: "RAG 是一种技术..." ← 不相关

重排模型对比：

本地 Reranker（BGE-Reranker）
├─ 用时：10-50ms（GPU 加速）
├─ 成本：0（开源免费）
├─ 精度：0.88 M-NDCG@10
└─ 推荐：自部署环境

API 型 Reranker（Cohere Rerank）
├─ 用时：500-1000ms（网络延迟）
├─ 成本：$3 / 1M API calls
├─ 精度：0.92 M-NDCG@10（更好）
└─ 推荐：追求最高精度
```

---

#### 第 6 步：缓存结果（Redis）

```
命令：redis.set(
          "search:6a9f8e2c8d4b1a5f...",
          json.dumps(results),
          ex=300  # ← 5 分钟后自动删除
      )

存的数据大小：通常 10-50KB（10 个 chunks × 1-5KB）

[Redis 写入用时：2ms]
```

**缓存生命周期**：

```
时间轴（ex=300 秒 = 5 分钟）：

00:00:00 - 用户问第 1 次 → 缓存 MISS → 搜索 → 存入 Redis
00:00:02 - 用户问第 2 次 → 缓存 HIT → 直接返回（2ms）
00:02:30 - 用户问第 3 次 → 缓存 HIT → 直接返回（2ms）
00:05:01 - Redis 自动删除缓存
00:05:02 - 用户问第 4 次 → 缓存 MISS → 搜索 → 重新存入 Redis
```

**缓存数据结构**：

```
Redis 中存的数据：

Key:   search:6a9f8e2c8d4b1a5f...
Value: {
           "query": "什么是 RAG？",
           "kb_ids": ["kb_123", "kb_456"],
           "top_k": 10,
           "results": [
               {
                   "chunk_id": "chunk_123",
                   "content": "RAG 是检索增强...",
                   "score": 0.885
               },
               ...
           ],
           "cached_at": 1730548845,  # Unix timestamp
           "ttl": 300  # 剩余过期时间（秒）
       }
TTL:   300 秒（5 分钟）

Redis 内存占用：
├─ 单次搜索结果：~30KB
├─ 热门搜索（缓存 100 个问题）：~3MB
├─ 占比：相对于总数据，极小
└─ 成本：极低
```

---

#### 第 7 步：返回结果给用户

```json
{
    "code": 0,
    "message": "success",
    "data": {
        "chunks": [
            {
                "chunk_id": "chunk_123",
                "content": "RAG 是检索增强生成...",
                "doc_id": "doc_456",
                "similarity": 0.885,
                "page_num": 5,
                "metadata": {
                    "source": "introduction.pdf"
                }
            },
            // ... 最多 10 条
        ],
        "total": 10,
        "used_tokens": 245,
        "response_time_ms": 87
    }
}
```

**总耗时计算**：

```
缓存命中（30-50% 情况）：
└─ 2ms（Redis 查询）= 总计 2ms

缓存未命中 + 不用重排（50-70% 情况）：
├─ 10-50ms（生成向量）
├─ + 50-200ms（Elasticsearch 搜索）
├─ + 2ms（Redis 缓存）
└─ = 62-252ms，平均 100-150ms

缓存未命中 + 使用重排（20-30% 情况）：
├─ 10-50ms（生成向量）
├─ + 50-200ms（Elasticsearch 搜索）
├─ + 100-500ms（Reranker）
├─ + 2ms（Redis 缓存）
└─ = 162-752ms，平均 300-400ms

用户体验目标：
├─ 缓存命中：2ms ✓ 瞬间响应
├─ 标准搜索：100-200ms ✓ 快速响应
├─ 带重排：300-400ms ~ 可接受
└─ > 1000ms ✗ 用户感觉慢
```

---

## 系统架构数据流

### 完整端到端流程

```
┌─────────────────────────────────────────────────────────────┐
│ 【用户交互层】                                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Web UI / Mobile App / API                                 │
│  • 上传文档                                                 │
│  • 提问问题                                                 │
│  • 查看历史                                                 │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   上传文档   │ │   提问查询   │ │  查看历史    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌──────────────────────────────────────────────────────────────┐
│ 【应用层】RAGFlow Backend                                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  API Routes & Business Logic                               │
│  • 文档处理 Service                                          │
│  • 搜索 Service                                              │
│  • 对话 Service                                              │
│  • 用户管理 Service                                          │
│                                                              │
└───────────┬─────────────┬──────────────┬─────────────┬───────┘
            │             │              │             │
   ┌────────▼─┐  ┌────────▼─┐  ┌────────▼─┐  ┌────────▼─┐
   │  文档    │  │  向量    │  │ 缓存/   │  │  数据    │
   │  处理    │  │  生成    │  │ 会话   │  │  持久化  │
   │          │  │          │  │        │  │          │
   └────────┬─┘  └────────┬─┘  └────────┬─┘  └────────┬─┘
            │             │              │             │
            ▼             ▼              ▼             ▼
┌──────────────────────────────────────────────────────────────┐
│ 【存储层】                                                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐    │
│  │  PostgreSQL  │   │ Elasticsearch│   │    Redis     │    │
│  │              │   │              │   │              │    │
│  │ • 元数据     │   │ • 文本向量   │   │ • 会话缓存   │    │
│  │ • 用户信息   │   │ • 倒排索引   │   │ • 查询结果   │    │
│  │ • 对话历史   │   │ • 全文搜索   │   │ • 计数器     │    │
│  │ • 配置信息   │   │              │   │              │    │
│  └──────────────┘   └──────────────┘   └──────────────┘    │
│                                                              │
│  ┌──────────────┐                                           │
│  │    MinIO     │                                           │
│  │              │                                           │
│  │ • 原始文件   │                                           │
│  │ • 处理中间文 │                                           │
│  │ • 缓存文件   │                                           │
│  └──────────────┘                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 性能分析

### 操作性能表现

| 操作 | 耗时 | 瓶颈 | 优化方法 |
|-----|------|------|---------|
| **Redis 查询** | 1-5ms | 网络往返 | 本地连接 |
| **向量生成** | 10-50ms | CPU/GPU 计算 | GPU 加速 |
| **ES 向量搜索** | 50-150ms | 索引查询 | 建立复合索引 |
| **Reranker** | 100-500ms | API 调用 | 本地部署 |
| **PostgreSQL 查询** | 10-100ms | 磁盘 I/O | 查询优化、缓存 |
| **MinIO 上传** | 5-60s | 网络、文件 I/O | 大文件分块上传 |
| **OCR 识别** | 2-3分钟 | CPU 密集 | GPU 加速 |

### 缓存效果分析

```
假设：100 位用户，每小时 1000 个查询

场景 A：无缓存
├─ 每个查询耗时：150ms（平均）
├─ 总耗时：1000 × 150ms = 150 秒
├─ CPU 使用率：高
├─ 向量搜索调用：1000 次
└─ ES 压力：大

场景 B：使用 Redis 缓存（5分钟）
├─ 缓存命中率：40%（热门问题反复问）
├─ 缓存命中查询：400 × 2ms = 0.8 秒
├─ 缓存未命中查询：600 × 150ms = 90 秒
├─ 总耗时：90.8 秒
├─ 时间节省：150 - 90.8 = 59.2 秒（39% 改进）
├─ ES 调用减少：400 次
└─ 压力：中等
```

---

## 故障排查指南

### 文档上传卡顿

```
问题：用户上传文档后，长时间显示"处理中"

可能原因：
├─ MinIO 满了（磁盘空间不足）
├─ 后台任务队列堆积
├─ OCR 服务崩溃
├─ 向量生成服务故障
└─ Elasticsearch 写入缓慢

排查步骤：
1. 检查 PostgreSQL documents 表的 progress_msg
   SELECT * FROM documents WHERE run = 1 ORDER BY updated_time DESC;
   → 看是否有错误信息

2. 检查后台任务日志
   docker logs ragflow-worker

3. 检查磁盘空间
   df -h /minio_storage

4. 检查 Elasticsearch 状态
   curl http://elasticsearch:9200/_cluster/health
   → 应该返回 "status": "green"
```

### 搜索结果不准确

```
问题：搜索结果与问题不相关

可能原因：
├─ 向量权重设置不当
├─ Reranker 未启用
├─ 向量模型不适配（英文模型搜中文）
├─ 知识库数据质量差
└─ 阈值设置过低

排查步骤：
1. 调整权重
   更改 vector_similarity_weight 从 0.3 → 0.5

2. 启用 Reranker
   添加 rerank_id 配置

3. 检查向量模型
   看知识库的 embedding_model 是否与内容语言匹配

4. 检查搜索结果的相似度分数
   应该 > 0.5（过低表示结果不相关）
```

### 缓存未生效

```
问题：同一个问题查询，第二次还是很慢

可能原因：
├─ Redis 连接故障
├─ 缓存键生成不一致
├─ 缓存过期时间太短
└─ 问题文本略有不同（空格、标点）

排查步骤：
1. 检查 Redis 连接
   redis-cli ping
   → 应该返回 PONG

2. 检查缓存键
   redis-cli KEYS "search:*"
   → 应该有数据

3. 延长缓存时间
   修改 cache_ttl 从 300 → 3600

4. 规范化问题文本
   去除多余空格和标点
```

---

## 最佳实践建议

### 文档管理
- 定期清理 MinIO 过期文件（OCR 结果、临时向量）
- 为原始文档设置备份
- 监控 MinIO 磁盘使用率

### 搜索优化
- 启用 Reranker 以获得最佳精度
- 根据内容调整向量权重
- 定期分析搜索日志，优化知识库

### 缓存管理
- 根据热数据情况调整 TTL
- 监控 Redis 内存使用
- 定期清理过期缓存

### 性能监控
- 监控 Elasticsearch 索引大小
- 跟踪平均搜索耗时
- 记录缓存命中率
- 设置告警阈值

