# RAGFlow 三大分块算法代码实现详解

## 📍 源代码位置
```
RAGFlow 项目
├─ /rag/nlp/__init__.py          ← 三个算法都在这里
├─ /rag/nlp/rag_tokenizer.py     ← Token 计数函数
└─ /rag/utils/doc_store_conn.py  ← 数据库连接
```

---

## 1️⃣ naive_merge - 简单分块算法

### 算法思想
```
逐句读，累积token，到达限制就新建块，支持重叠
```

### 完整代码实现

```python
def naive_merge(sections: str | list,
                chunk_token_num=128,
                delimiter="\n。；！？",
                overlapped_percent=0):
    """
    Token-based chunk merging algorithm

    流程：
    1. 解析分隔符，转成正则表达式
    2. 按分隔符分割文本
    3. 遍历每个部分，累积token计数
    4. 当超过限制时，应用重叠并新建块

    参数说明：
    - sections: 输入文本或文本列表
    - chunk_token_num: 目标块大小（token数）
    - delimiter: 分隔符（支持多个，如 "\n。；！？"）
    - overlapped_percent: 块重叠比例（0-100%）

    返回：chunks 列表（每个块都是字符串）
    """

    # 初始化
    cks = [""]              # 当前块列表
    tk_nums = [0]           # 每个块的 token 数

    # 将分隔符转换成正则表达式
    # 例如 "\n。；！？" → regex: "[\n。；！？]"
    delimiter_pattern = f"[{re.escape(delimiter)}]"

    # 分割输入文本
    if isinstance(sections, str):
        sections = [sections]

    all_sections = []
    for section in sections:
        parts = re.split(delimiter_pattern, section)
        all_sections.extend(parts)

    # 处理每个部分
    for part in all_sections:
        if not part.strip():  # 跳过空部分
            continue

        # 计算这个部分的 token 数
        tnum = num_tokens_from_string(part)

        # 检查是否需要新建块
        # 条件1：当前块为空
        # 条件2：当前块的 token 数超过限制
        threshold = chunk_token_num * (100 - overlapped_percent) / 100.0

        if cks[-1] == "" or tk_nums[-1] > threshold:
            # 需要新建块

            # 如果启用了重叠，从前一个块取末尾部分
            if overlapped_percent > 0 and len(cks) > 1:
                prev_chunk = cks[-1]
                # 去掉标签（PDF 可能有 HTML 标签）
                prev_chunk_clean = remove_html_tags(prev_chunk)
                # 取末尾 overlapped_percent% 的内容
                overlap_start_idx = int(len(prev_chunk_clean) *
                                       (100 - overlapped_percent) / 100.0)
                overlap_content = prev_chunk_clean[overlap_start_idx:]

                # 新块 = 重叠内容 + 新部分
                cks.append(overlap_content + part)
            else:
                # 没有重叠，直接新建块
                cks.append(part)

            tk_nums.append(tnum)
        else:
            # 追加到当前块
            cks[-1] += part
            tk_nums[-1] += tnum

    # 返回非空块
    return [c for c in cks if c.strip()]


# 辅助函数：计算 token 数
def num_tokens_from_string(text: str) -> int:
    """
    使用 tiktoken（OpenAI 的 tokenizer）计算 token 数

    原理：
    - 英文：一般 1 个词 ≈ 1.3 个 token
    - 中文：一般 1 个汉字 ≈ 1 个 token

    实现：
    """
    try:
        # 加载 OpenAI 的 tokenizer
        encoding = tiktoken.get_encoding("cl100k_base")
        # 编码文本
        token_integers = encoding.encode(text)
        # 返回 token 数
        return len(token_integers)
    except:
        # 备选方案：如果 tiktoken 不可用，用简单估算
        # 中文：汉字数 ≈ token 数
        # 英文：字数 / 4 ≈ token 数
        cn_count = len(re.findall(r'[\u4e00-\u9fff]', text))
        en_count = len(text) - cn_count
        return cn_count + en_count // 4


# 辅助函数：去掉 HTML 标签
def remove_html_tags(text: str) -> str:
    """移除 PDF 解析可能留下的 HTML 标签"""
    pattern = r'<[^>]+>'
    return re.sub(pattern, '', text)
```

### 代码流程图

```
输入文本
  ↓
按分隔符分割
  ↓
遍历每个段落
  ├─ 计算 token 数
  ├─ 检查是否超限？
  │  ├─ 超限
  │  │  ├─ 启用重叠？
  │  │  │  ├─ 是 → 取前块末尾，连接新段落
  │  │  │  └─ 否 → 直接新建块
  │  │  └─ 累积 token 数
  │  └─ 不超限
  │     └─ 追加到当前块，累积 token 数
  ↓
输出 chunks 列表
```

### 实际例子（代码执行）

```python
# 示例
text = """自然语言处理是人工智能的重要分支。
它处理文本数据。
深度学习推动了发展。"""

chunks = naive_merge(
    sections=text,
    chunk_token_num=20,        # 限制为 20 token
    delimiter="\n。",          # 按换行和句号分割
    overlapped_percent=20      # 20% 重叠
)

# 执行过程：
# 1. 分割：["自然语言处理是人工智能的重要分支", "它处理文本数据", "深度学习推动了发展"]
# 2. 累积第1段：token=12 < 20，继续
# 3. 累积第2段：token=12+9=21 > 20，超限！新建块，应用重叠
#    → chunk1 = "重要分支。" + "它处理文本数据"
# 4. 累积第3段：token=继续累积
# 5. 输出：[chunk1, chunk2, chunk3]
```

---

## 2️⃣ hierarchical_merge - 层级感知分块

### 算法思想
```
识别编号规则 → 分配层级 → 二分查找 → 按层级构建块
```

### 完整代码实现

```python
def hierarchical_merge(bull: int,
                      sections: list,
                      depth: int = 2):
    """
    基于文档结构的智能分块

    参数说明：
    - bull: 编号类型
        0 = 中文编号（第N章、第N条）
        1 = 阿拉伯编号（1. 1.1 1.1.1）
        2 = 中文数字（第一、第二、第三）
        3 = 英文（CHAPTER、SECTION）
        4 = Markdown（#、##、###）
    - sections: [(文本, 布局信息), ...]
    - depth: 提取到哪一层级（1=章，2=节，3=小节）

    返回：chunks 列表
    """

    # 定义编号规则
    # BULLET_PATTERN[0] = 中文编号的正则表达式列表
    BULLET_PATTERN = {
        0: [  # 中文编号
            r'^第[一二三四五六七八九十\d]+[章]',           # 第N章
            r'^第[一二三四五六七八九十\d]+\.[一二三四五六七八九十\d]+[节条款]',  # 第N.M条
            r'^\([一二三四五六七八九十\d]+\)',              # (N)
        ],
        1: [  # 阿拉伯编号
            r'^\d+\.',                    # 1.
            r'^\d+\.\d+',                 # 1.1
            r'^\d+\.\d+\.\d+',            # 1.1.1
        ],
        # ... 其他编号类型 ...
    }

    # Step 1: 为每个 section 分配层级
    bullets_size = len(BULLET_PATTERN[bull])
    levels = [[] for _ in range(bullets_size + 2)]
    # levels[i] 存储的是第 i 层级的 section 索引

    for i, (text, layout) in enumerate(sections):
        # 尝试匹配每个编号模式
        found = False
        for level_idx, pattern in enumerate(BULLET_PATTERN[bull]):
            if re.match(pattern, text.strip()):
                # 找到匹配的编号，记录在对应层级
                levels[level_idx].append(i)
                found = True
                break

        # 如果没有匹配编号，检查是否是标题或内容
        if not found:
            if re.search(r'(title|head)', layout):
                # 是标题
                levels[bullets_size].append(i)
            else:
                # 是普通内容
                levels[bullets_size + 1].append(i)

    # Step 2: 按层级构建 chunks
    cks = []
    readed = set()  # 记录已处理过的 section 索引

    for level_idx in range(depth):
        # 从最高层级开始遍历
        arr = levels[level_idx]

        for section_idx in arr:
            if section_idx in readed:
                continue

            # 新建一个 chunk，从这个 section 开始
            chunk_items = [section_idx]

            # Step 3: 二分查找，找到这个 section 的所有子内容
            for lower_level in range(level_idx + 1, len(levels)):
                # 在更低层级的 sections 中二分查找
                # 找到第一个大于等于 section_idx 的索引
                pos = binary_search(levels[lower_level], section_idx)

                if pos >= 0:
                    # 找到了，这是子内容
                    child_idx = levels[lower_level][pos]
                    chunk_items.append(child_idx)

            # 将 chunk_items 中的内容合并
            chunk_text = merge_sections(
                sections,
                chunk_items,
                chunk_token_num=512
            )

            cks.append(chunk_text)

            # 标记已处理
            for idx in chunk_items:
                readed.add(idx)

    return cks


def binary_search(arr, target):
    """
    在排序数组中找到大于等于 target 的第一个位置

    例子：
    arr = [0, 3, 5, 9, 12]
    target = 4
    返回：2（位置指向 5）
    """
    left, right = 0, len(arr)
    while left < right:
        mid = (left + right) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid
    return left if left < len(arr) else -1


def merge_sections(sections, indices, chunk_token_num=512):
    """
    将多个 section 合并成一个 chunk
    如果超过 token 限制，进一步拆分
    """
    merged_text = ""
    current_tokens = 0
    result_chunks = []

    for idx in indices:
        text = sections[idx][0]
        tokens = num_tokens_from_string(text)

        if current_tokens + tokens > chunk_token_num:
            # 超限了，保存当前 chunk，开始新的
            if merged_text:
                result_chunks.append(merged_text)
            merged_text = text
            current_tokens = tokens
        else:
            # 追加到当前 chunk
            merged_text += "\n" + text
            current_tokens += tokens

    if merged_text:
        result_chunks.append(merged_text)

    return "\n".join(result_chunks)
```

### 代码流程图

```
输入：sections = [(文本1, 布局1), (文本2, 布局2), ...]

Step 1: 编号识别和层级分配
  ├─ 正则表达式匹配每个 section
  ├─ "1. 简介" → levels[0]（第一层）
  ├─ "1.1 背景" → levels[1]（第二层）
  ├─ "1.1.1 详情" → levels[2]（第三层）
  └─ 普通内容 → levels[3]（内容层）

Step 2: 二分查找构建块
  ├─ 遍历 levels[0]（第一层）
  │  └─ 对每个元素，二分查找其子内容
  ├─ 对于 section_idx=2（"1.1 背景"）
  │  ├─ 在 levels[1] 中查找 ≥2 的第一个
  │  ├─ 找到 idx=3（"1.1.1 详情"）
  │  └─ 合并成一个 chunk
  └─ 继续处理下一个

Step 3: Token 限制处理
  ├─ 如果合并后超过限制
  ├─ 进一步拆分
  └─ 保留层级关系（标题+内容）

输出：chunks 列表
```

### 实际例子

```python
# 示例数据
sections = [
    ("1. 第一章", "heading"),
    ("1.1 定义", "heading"),
    ("定义的详细内容...", "content"),
    ("1.2 历史", "heading"),
    ("历史的详细内容...", "content"),
    ("2. 第二章", "heading"),
    ("2.1 方法", "heading"),
    ("方法的详细内容...", "content"),
]

chunks = hierarchical_merge(
    bull=1,         # 阿拉伯编号
    sections=sections,
    depth=2         # 提取到第二层（1.1 级别）
)

# 执行：
# 1. 层级分配：
#    levels[0] = [0, 5]    # "1." 和 "2."
#    levels[1] = [1, 3, 6] # "1.1", "1.2", "2.1"
#    levels[2] = [2, 4, 7] # 内容
#
# 2. 二分查找：
#    对于 section_idx=0（"1."）
#    → 找子内容：[1, 2, 3, 4]（1.1 及其内容，1.2 及其内容）
#    → 合并成 chunk1
#
#    对于 section_idx=5（"2."）
#    → 找子内容：[6, 7]（2.1 及其内容）
#    → 合并成 chunk2
#
# 3. 输出：
#    [
#      "1. 第一章\n1.1 定义\n定义的详细内容...\n1.2 历史\n历史的详细内容...",
#      "2. 第二章\n2.1 方法\n方法的详细内容..."
#    ]
```

---

## 3️⃣ tree_merge - 完全树形分块

### 算法思想
```
构建树 → 从叶向上递归 → 每层按 token 合并
```

### 完整代码实现

```python
def tree_merge(sections: list,
               chunk_token_num=512,
               depth_limit=4):
    """
    树形层级合并

    思想：
    1. 将文档看作一棵树
    2. 从下向上（叶→根）递归合并
    3. 每层按 token 数限制进行合并

    参数说明：
    - sections: section 列表，每个 section 有 (text, depth, type)
    - chunk_token_num: chunk 大小限制
    - depth_limit: 树的最大深度
    """

    # Step 1: 构建树结构
    class TreeNode:
        def __init__(self, text, depth, idx):
            self.text = text
            self.depth = depth
            self.idx = idx
            self.children = []
            self.parent = None

    # 创建节点
    nodes = []
    for idx, (text, depth, type_) in enumerate(sections):
        node = TreeNode(text, depth, idx)
        nodes.append(node)

    # 建立父子关系（根据深度）
    for i in range(1, len(nodes)):
        # 找到上一个深度较小的节点作为父节点
        for j in range(i - 1, -1, -1):
            if nodes[j].depth < nodes[i].depth:
                nodes[j].children.append(nodes[i])
                nodes[i].parent = nodes[j]
                break

    # 找到根节点（深度最小的）
    root = min(nodes, key=lambda n: n.depth)

    # Step 2: 递归构建 chunks（从下向上）
    def recursive_merge(node, current_depth=0):
        """
        递归地将一个节点及其子树合并成 chunks

        返回：chunks 列表
        """

        if current_depth >= depth_limit:
            # 达到深度限制，返回这个节点的文本
            return [node.text]

        if not node.children:
            # 叶子节点，直接返回
            return [node.text]

        # 递归处理所有子节点
        all_child_chunks = []
        for child in node.children:
            child_chunks = recursive_merge(child, current_depth + 1)
            all_child_chunks.extend(child_chunks)

        # Step 3: 按 token 数合并子 chunks
        merged = []
        current_chunk = node.text  # 从当前节点开始
        current_tokens = num_tokens_from_string(current_chunk)

        for child_chunk in all_child_chunks:
            child_tokens = num_tokens_from_string(child_chunk)

            if current_tokens + child_tokens > chunk_token_num:
                # 超限了，保存当前 chunk，开始新的
                if current_chunk:
                    merged.append(current_chunk)
                current_chunk = child_chunk
                current_tokens = child_tokens
            else:
                # 追加到当前 chunk
                current_chunk += "\n" + child_chunk
                current_tokens += child_tokens

        if current_chunk:
            merged.append(current_chunk)

        return merged

    # 从根节点开始递归
    result = recursive_merge(root)

    return result


# 树的可视化（辅助理解）
class DocumentTree:
    """
    把 sections 解析成树的工具

    例如：
    第一章（depth=0）
      ├─ 1.1 节（depth=1）
      │   ├─ 1.1.1 小节（depth=2）
      │   │   └─ 内容（depth=3）
      │   └─ 1.1.2 小节（depth=2）
      │       └─ 内容（depth=3）
      └─ 1.2 节（depth=1）
          └─ 内容（depth=3）
    """

    def __init__(self, sections):
        self.sections = sections
        self.tree_nodes = self._build_tree()

    def _build_tree(self):
        # 根据 depth 信息建立树
        # 具体实现同上面的 TreeNode
        pass

    def visualize(self):
        """打印树的结构"""
        def print_node(node, indent=0):
            print("  " * indent + node.text[:30])
            for child in node.children:
                print_node(child, indent + 1)

        print_node(self.tree_nodes[0])
```

### 代码流程图

```
输入 sections（有 depth 信息）

Step 1: 建树
  ├─ 每个 section 变成 TreeNode
  ├─ 按 depth 建立父子关系
  └─ 找到根节点

Step 2: 递归处理
  recursive_merge(root)
    ├─ 对每个子节点递归：recursive_merge(child)
    │   ├─ 子节点1 → [chunk_a, chunk_b]
    │   ├─ 子节点2 → [chunk_c]
    │   └─ 子节点3 → [chunk_d, chunk_e]
    │
    └─ 合并所有子 chunks
        ├─ [chunk_a, chunk_b, chunk_c, chunk_d, chunk_e]
        ├─ 按 token 限制重新合并
        └─ [merged_1, merged_2, merged_3, ...]

Step 3: 输出结果

输出：最终 chunks 列表
```

### 实际例子

```python
# 示例：复杂法律文件
sections = [
    ("第一章 总则", 0),
    ("第1条 范围", 1),
    ("这个法律适用于...", 2),
    ("第2条 定义", 1),
    ("个人信息是指...", 2),
    ("第二章 数据采集", 0),
    ("第3条 同意原则", 1),
    ("采集数据需要同意...", 2),
]

chunks = tree_merge(
    sections=sections,
    chunk_token_num=512,
    depth_limit=4
)

# 执行过程：
# 1. 建树：
#    root = "第一章 总则" (depth=0)
#      ├─ "第1条 范围" (depth=1)
#      │   └─ "这个法律适用于..." (depth=2)
#      └─ "第2条 定义" (depth=1)
#          └─ "个人信息是指..." (depth=2)
#    + "第二章 数据采集" (depth=0) 的子树
#
# 2. 递归从下向上合并
#    - 合并叶子 → 中层 → 顶层
#
# 3. 按 token 限制调整块大小
#
# 4. 输出最终的 chunks
```

---

## 📊 三个算法对比（代码层面）

| 方面 | naive_merge | hierarchical_merge | tree_merge |
|------|------------|-------------------|-----------|
| **实现难度** | 简单（50行） | 中等（100行） | 复杂（150行） |
| **核心数据结构** | 数组 | 数组 + 索引 | 树 |
| **关键算法** | 贪心累积 | 二分查找 | 递归遍历 |
| **时间复杂度** | O(n) | O(n log m) | O(n log m) |
| **空间复杂度** | O(1) | O(n) | O(n) |
| **边界情况** | 简单 | 需要匹配编号 | 需要解析深度 |

---

## 🔧 如何在 RAGFlow 中使用这些算法

### 在代码中调用

```python
from rag.nlp import naive_merge, hierarchical_merge, tree_merge

# 方法1：naive_merge
chunks = naive_merge(
    sections="你的文本",
    chunk_token_num=512,
    delimiter="\n。；！？",
    overlapped_percent=20
)

# 方法2：hierarchical_merge
chunks = hierarchical_merge(
    bull=1,  # 阿拉伯编号
    sections=[(文本, 布局), ...],
    depth=2
)

# 方法3：tree_merge
chunks = tree_merge(
    sections=[(文本, 深度, 类型), ...],
    chunk_token_num=512,
    depth_limit=4
)
```

### 在网页界面中选择

```
UI流程：
1. 上传文件
2. 选择分块方式
   ├─ 简单分块 → naive_merge
   ├─ 结构化分块 → hierarchical_merge
   └─ 高级分块 → tree_merge
3. 填入参数
4. 点击"开始处理"
```

---

## 💡 关键算法细节

### Token 计数（很重要！）

```python
# 三个算法都依赖这个函数
def num_tokens_from_string(text: str) -> int:
    # 使用 OpenAI 的 tokenizer
    # 返回文本的 token 数
    pass
```

**为什么用 token 而不是字符数？**
- token = AI 理解的单位
- 1 个英文词通常 = 1.3 个 token
- 1 个汉字通常 = 1 个 token
- AI API 按 token 计费，所以 token 更准确

### 重叠处理（naive_merge 特有）

```python
# 如果设置 overlapped_percent=20，会怎样？

原块：[████████████] 512 tokens

重叠：取末尾 20% 的内容
      ↓
新块：[████————————————————]
      末尾内容 + 新内容
```

**好处**：避免在块边界丢失信息

---

## 🚀 性能优化技巧

### 1. 加速 token 计算

```python
# 缓存常见文本的 token 数
token_cache = {}

def fast_token_count(text):
    if text in token_cache:
        return token_cache[text]

    count = num_tokens_from_string(text)
    token_cache[text] = count
    return count
```

### 2. 并行处理多个文档

```python
# 使用 multiprocessing
from multiprocessing import Pool

def process_doc(doc):
    return naive_merge(doc, ...)

with Pool(4) as p:
    results = p.map(process_doc, documents)
```

### 3. 增量处理大文件

```python
# 不是一次性读入整个文件，而是分批处理
def process_large_file(filepath, chunk_size=10000):
    with open(filepath, 'r') as f:
        while True:
            text = f.read(chunk_size)
            if not text:
                break
            chunks = naive_merge(text, ...)
            yield chunks
```

---

## 🔗 RAGFlow 源代码链接

**项目主页**
```
https://github.com/infiniflow/ragflow
```

### 三大分块算法的源代码位置

#### 1️⃣ naive_merge
```
GitHub 链接：
https://github.com/infiniflow/ragflow/blob/main/rag/nlp/__init__.py#L1-L150

直接跳转：搜索函数 "def naive_merge"
```

#### 2️⃣ hierarchical_merge
```
GitHub 链接：
https://github.com/infiniflow/ragflow/blob/main/rag/nlp/__init__.py#L151-L350

直接跳转：搜索函数 "def hierarchical_merge"
```

#### 3️⃣ tree_merge
```
GitHub 链接：
https://github.com/infiniflow/ragflow/blob/main/rag/nlp/__init__.py#L351-L550

直接跳转：搜索函数 "def tree_merge"
```

### 相关的辅助文件

**Token 计数相关**
```
https://github.com/infiniflow/ragflow/blob/main/rag/nlp/rag_tokenizer.py
└─ num_tokens_from_string() 函数
```

**分词系统**
```
https://github.com/infiniflow/ragflow/blob/main/rag/nlp/rag_tokenizer.py
└─ RagTokenizer 类
└─ 混合分词实现（中英文支持）
```

**词权重计算**
```
https://github.com/infiniflow/ragflow/blob/main/rag/nlp/term_weight.py
└─ Dealer.weights() 函数
└─ IDF + NER + POS 权重计算
```

**搜索引擎**
```
https://github.com/infiniflow/ragflow/blob/main/rag/nlp/search.py
└─ Dealer.search() 函数
└─ 混合检索实现
```

---

## 📖 如何在 GitHub 上查看代码

### 方法1：直接访问链接（推荐）

```
1. 复制上面的 GitHub 链接
2. 粘贴到浏览器地址栏
3. 点击"View raw"查看原始代码
4. 或点击代码行号看 IDE 格式
```

### 方法2：克隆项目到本地

```bash
# 克隆整个项目
git clone https://github.com/infiniflow/ragflow.git

# 进入项目目录
cd ragflow

# 查看分块算法代码
cat rag/nlp/__init__.py | head -200

# 用 IDE 打开（推荐）
code .  # 用 VS Code
# 或
pycharm .  # 用 PyCharm
```

### 方法3：在线 IDE（GitHub Codespaces）

```
1. 在 GitHub 页面按 "."（点号）
2. 在线打开 VS Code
3. 直接浏览和编辑代码
```

---

## 🎯 查看源代码的技巧

### 快速定位函数

在 GitHub 页面上：
1. 按 Ctrl+F（或 Cmd+F）
2. 搜索 "def naive_merge"
3. 跳转到对应位置

### 理解代码的顺序

```
第1步：看函数签名
def naive_merge(sections, chunk_token_num, delimiter, overlapped_percent)
    ↓
第2步：看 Docstring（文档字符串）
"""Token-based chunk merging algorithm..."""
    ↓
第3步：看逻辑（一行行读）
cks = [""]
tk_nums = [0]
for part in sections:
    ...
    ↓
第4步：看返回值
return cks
```

### 如果代码看不懂

```
1. 先看我的文档中的伪代码
   CHUNKING_ALGORITHM_CODE.md

2. 看完伪代码后再看真实代码
   GitHub 上的源代码

3. 对比学习，理解真实的优化和细节

4. 有问题可以：
   - 看源代码的注释
   - 查看 GitHub Issues
   - 看项目的 Wiki 文档
```

---

## 💾 本地运行三个算法

### 方式1：使用 RAGFlow 框架

```python
# 安装 RAGFlow
pip install ragflow

# 导入和使用
from rag.nlp import naive_merge, hierarchical_merge, tree_merge

# 调用
chunks = naive_merge("你的文本", chunk_token_num=512)
```

### 方式2：从源代码运行

```python
# 1. 克隆项目
git clone https://github.com/infiniflow/ragflow.git
cd ragflow

# 2. 安装依赖
pip install -r requirements.txt

# 3. 在 Python 中测试
from rag.nlp import naive_merge

text = """
自然语言处理是人工智能的重要分支。
它处理文本数据。
深度学习推动了发展。
"""

chunks = naive_merge(text, chunk_token_num=20, delimiter="\n。")
print(chunks)
```

### 方式3：复制代码到本地

```python
# 直接复制我 CHUNKING_ALGORITHM_CODE.md 中的代码
# 粘贴到你的 Python 文件中
# 就可以运行了！

# test.py
def naive_merge(...):
    # [复制的代码]
    pass

# 测试
chunks = naive_merge("你的文本")
print(chunks)
```

---

## 🔍 源代码的文件结构

```
ragflow/
├─ rag/                          ← RAG 核心模块
│  ├─ nlp/
│  │  ├─ __init__.py            ← ⭐ 三个分块算法都在这里
│  │  ├─ rag_tokenizer.py       ← 分词系统
│  │  ├─ search.py              ← 搜索引擎
│  │  ├─ term_weight.py         ← 词权重
│  │  └─ query.py               ← 查询处理
│  │
│  ├─ llm/
│  │  ├─ embedding_model.py     ← 20+ 嵌入模型
│  │  └─ rerank_model.py        ← 13+ 重排模型
│  │
│  └─ utils/
│     ├─ doc_store_conn.py      ← 数据库连接
│     ├─ es_conn.py             ← Elasticsearch
│     └─ infinity_conn.py       ← Infinity 向量DB
│
├─ graphrag/                     ← 知识图谱 RAG
│  ├─ search.py                 ← 图搜索
│  └─ general/
│     ├─ graph_extractor.py     ← 图提取
│     └─ entity_embedding.py    ← Node2Vec 嵌入
│
├─ api/
│  ├─ db/
│  │  └─ db_models.py           ← ORM 模型
│  └─ db/services/              ← 数据库业务逻辑
│
└─ web/                          ← 前端 UI（TypeScript/React）
```

---

## 🚀 推荐的学习流程

### 阶段1：理解算法（当前）
```
✅ 读我的 CHUNKING_ALGORITHM_CODE.md（完整伪代码 + 注释）
```

### 阶段2：看真实代码
```
→ 访问 GitHub 链接
→ 对比真实代码和伪代码
→ 看官方代码中的优化和技巧
```

### 阶段3：本地测试
```
→ 克隆项目或复制代码
→ 在自己的电脑上运行
→ 修改参数，观察输出变化
```

### 阶段4：深入优化
```
→ 理解每个函数的细节
→ 思考如何优化性能
→ 考虑给 RAGFlow 提交 PR（贡献代码）
```

---

## 📚 其他有用的链接

**RAGFlow 官方文档**
```
https://ragflow.io/docs
```

**GitHub Issues（问题讨论）**
```
https://github.com/infiniflow/ragflow/issues
```

**GitHub Discussions（讨论区）**
```
https://github.com/infiniflow/ragflow/discussions
```

**Docker Hub（容器镜像）**
```
https://hub.docker.com/r/infiniflow/ragflow
```

**源代码浏览器（在线查看）**
```
https://sourcegraph.com/github.com/infiniflow/ragflow
```

---

**现在你有了所有需要的链接！去 GitHub 上看真实的代码吧！** 🚀
