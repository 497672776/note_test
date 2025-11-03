# RAGFlow API 架构分析

## 1. 系统启动流程

### 1.1 主入口
- **启动脚本**: `/docker/entrypoint.sh`
- **API服务启动**: `api/ragflow_server.py` (python3)
- **监听地址**: 由环境变量 `HOST_IP` 和 `HOST_PORT` 决定（默认: 0.0.0.0:9380）

### 1.2 启动组件
启动脚本支持以下模块的启动与管理：

| 组件 | 启动命令 | 说明 |
|-----|--------|------|
| Web Server | nginx + ragflow_server | 主API服务 |
| Admin Server | admin/server/admin_server.py | 管理后台 |
| MCP Server | mcp/server/server.py | MCP协议支持 |
| Task Executor | rag/svr/task_executor.py | 后台任务执行器 |

## 2. API 框架架构

### 2.1 核心框架
- **框架**: Flask (Werkzeug WSGI)
- **配置文件**: `/api/apps/__init__.py`
- **自动注册机制**: 动态加载 `*_app.py` 文件作为蓝图

### 2.2 API 文档
- **Swagger UI**: `/apidocs/`
- **API 规范**: `/apispec.json`
- **标题**: RAGFlow API v1.0.0
- **认证方式**: API Key (Authorization header)

### 2.3 中间件与安全
- **CORS**: 启用 (credentials=True, max_age=2592000)
- **会话**: Flask-Session (filesystem)
- **认证**: Flask-Login + JWT
- **邮件**: Flask-Mail (可选SMTP配置)
- **请求大小限制**: 可通过 `MAX_CONTENT_LENGTH` 设置（默认1GB）

## 3. API 应用模块

### 3.1 系统级应用
| 模块 | 路由前缀 | 功能 |
|-----|--------|------|
| user_app.py | `/v1/user` | 用户管理、登录、权限 |
| system_app.py | `/v1/system` | 系统配置、版本信息 |
| tenant_app.py | `/v1/tenant` | 租户管理 |
| llm_app.py | `/v1/llm` | LLM模型配置 |
| plugin_app.py | `/v1/plugin` | 插件管理 |

### 3.2 数据与知识库应用
| 模块 | 路由前缀 | 功能 |
|-----|--------|------|
| kb_app.py | `/v1/knowledge_base` | 知识库创建、更新、删除 |
| document_app.py | `/v1/document` | 文档管理 |
| chunk_app.py | `/v1/chunk` | 文本分块管理 |
| file_app.py | `/v1/file` | 文件上传、管理 |
| file2document_app.py | `/v1/file2document` | 文件到文档的转换 |

### 3.3 对话与交互应用
| 模块 | 路由前缀 | 功能 |
|-----|--------|------|
| conversation_app.py | `/v1/conversation` | 对话历史管理 |
| dialog_app.py | `/v1/dialog` | 聊天应用配置 |
| canvas_app.py | `/v1/canvas` | 工作流画布 |

### 3.4 其他应用
| 模块 | 路由前缀 | 功能 |
|-----|--------|------|
| search_app.py | `/v1/search` | 搜索应用管理 |
| api_app.py | `/v1/api` | API密钥与开放API |
| mcp_server_app.py | `/v1/mcp_server` | MCP服务器配置 |
| langfuse_app.py | `/v1/langfuse` | Langfuse集成 |

## 4. SDK API (外部集成)

### 4.1 路由前缀
所有SDK API均以 `/api/v1/` 为前缀（区别于内部API的 `/v1/`）

### 4.2 主要SDK端点

#### 4.2.1 数据集管理 (dataset.py)
```
POST   /api/v1/datasets              - 创建数据集
GET    /api/v1/datasets              - 列表查询
PUT    /api/v1/datasets/<dataset_id> - 更新数据集
DELETE /api/v1/datasets              - 删除数据集
GET    /api/v1/datasets/<dataset_id>/knowledge_graph     - 获取知识图谱
DELETE /api/v1/datasets/<dataset_id>/knowledge_graph     - 删除知识图谱
```

#### 4.2.2 文档管理 (doc.py)
```
POST   /api/v1/datasets/<dataset_id>/documents                    - 创建文档
GET    /api/v1/datasets/<dataset_id>/documents                    - 列表查询
PUT    /api/v1/datasets/<dataset_id>/documents/<document_id>      - 更新文档
GET    /api/v1/datasets/<dataset_id>/documents/<document_id>      - 文档详情
DELETE /api/v1/datasets/<dataset_id>/documents                    - 删除文档
POST   /api/v1/datasets/<dataset_id>/chunks                       - 创建分块
DELETE /api/v1/datasets/<dataset_id>/chunks                       - 删除分块
GET    /api/v1/datasets/<dataset_id>/documents/<document_id>/chunks - 获取分块
POST   /api/v1/datasets/<dataset_id>/documents/parse              - 解析文档
```

#### 4.2.3 检索与RAG (doc.py)
```
POST   /api/v1/retrieval             - RAG检索接口 (核心接口)
```

#### 4.2.4 聊天/对话 (chat.py)
```
POST   /api/v1/chats                 - 创建聊天应用
GET    /api/v1/chats                 - 列表查询
PUT    /api/v1/chats/<chat_id>       - 更新聊天应用
DELETE /api/v1/chats                 - 删除聊天应用
```

#### 4.2.5 会话与对话 (session.py)
```
POST   /api/v1/chats/<chat_id>/sessions                    - 创建会话
GET    /api/v1/chats/<chat_id>/sessions                    - 列表查询
PUT    /api/v1/chats/<chat_id>/sessions/<session_id>       - 更新会话
DELETE /api/v1/chats/<chat_id>/sessions                    - 删除会话

POST   /api/v1/chats/<chat_id>/completions                 - 聊天完成 (核心对话接口)
POST   /api/v1/chats_openai/<chat_id>/chat/completions     - OpenAI兼容接口
POST   /api/v1/agents_openai/<agent_id>/chat/completions   - Agent OpenAI兼容接口

POST   /api/v1/sessions/ask                                - 快速问答
POST   /api/v1/sessions/related_questions                  - 相关问题建议

POST   /api/v1/chatbots/<dialog_id>/completions            - 聊天机器人对话
GET    /api/v1/chatbots/<dialog_id>/info                   - 聊天机器人信息
```

#### 4.2.6 Agent (agent.py)
```
GET    /api/v1/agents                - 列表查询
POST   /api/v1/agents                - 创建Agent
PUT    /api/v1/agents/<agent_id>     - 更新Agent
DELETE /api/v1/agents/<agent_id>     - 删除Agent

POST   /api/v1/agents/<agent_id>/sessions          - 创建Agent会话
POST   /api/v1/agents/<agent_id>/completions       - Agent对话
GET    /api/v1/agents/<agent_id>/sessions          - 列表查询
DELETE /api/v1/agents/<agent_id>/sessions          - 删除会话

POST   /api/v1/agentbots/<agent_id>/completions    - Agent机器人对话
GET    /api/v1/agentbots/<agent_id>/inputs         - Agent输入定义
```

#### 4.2.7 文件管理 (files.py)
```
POST   /api/v1/file/upload           - 上传文件
POST   /api/v1/file/create           - 创建文件/文件夹
GET    /api/v1/file/list             - 列表查询
GET    /api/v1/file/root_folder      - 获取根文件夹
GET    /api/v1/file/parent_folder    - 获取父文件夹
GET    /api/v1/file/all_parent_folder - 获取所有父文件夹
POST   /api/v1/file/rm               - 删除文件
POST   /api/v1/file/rename           - 重命名文件
GET    /api/v1/file/get/<file_id>    - 获取文件
POST   /api/v1/file/mv               - 移动文件
POST   /api/v1/file/convert          - 转换文件
```

#### 4.2.8 搜索应用 (session.py)
```
POST   /api/v1/searchbots/ask                      - 搜索提问
POST   /api/v1/searchbots/retrieval_test           - 检索测试
POST   /api/v1/searchbots/related_questions        - 相关问题
GET    /api/v1/searchbots/detail                   - 搜索应用详情
POST   /api/v1/searchbots/mindmap                  - 思维导图
```

#### 4.2.9 第三方集成
```
POST   /api/v1/dify/retrieval        - Dify检索集成
```

## 5. 核心RAG相关API

### 5.1 RAG检索接口 (核心)
**路由**: `POST /api/v1/retrieval`
**功能**: 进行向量相似度检索，返回相关文档片段
**认证**: Token Required
**请求参数示例**:
```json
{
  "dataset_ids": ["kb_id1", "kb_id2"],
  "query": "用户问题",
  "top_n": 6,
  "top_k": 1024,
  "similarity_threshold": 0.2
}
```

### 5.2 聊天完成接口 (核心)
**路由**: `POST /api/v1/chats/<chat_id>/completions`
**功能**: 使用RAG增强的LLM生成回答
**认证**: Token Required
**请求参数示例**:
```json
{
  "session_id": "session_uuid",
  "messages": [{"role": "user", "content": "问题"}],
  "stream": true
}
```

### 5.3 知识库创建 (KB)
**路由**: `POST /v1/knowledge_base/create`
**功能**: 创建RAG数据集
**认证**: Login Required
**关键配置**:
- Parser: 文档解析器 (默认: naive)
- Parser Config:
  - layout_recognize: DeepDOC
  - chunk_token_num: 512
  - raptor: 层级摘要 (可选)
  - graphrag: 图谱提取 (可选)

## 6. 认证方式

### 6.1 内部API认证
- **方式**: Flask-Login + JWT Token
- **Header**: `Authorization: {jwt_token}`
- **装饰器**: `@login_required`

### 6.2 SDK API认证
- **方式**: Token Bearer
- **Header**: `Authorization: Bearer {api_token}`
- **装饰器**: `@token_required`

## 7. API 调用示例

### 7.1 创建聊天应用
```bash
curl -X POST http://localhost:9380/api/v1/chats \
  -H "Authorization: Bearer {api_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Chat",
    "dataset_ids": ["kb_id"],
    "llm": {"model_name": "gpt-4"}
  }'
```

### 7.2 进行对话
```bash
curl -X POST http://localhost:9380/api/v1/chats/{chat_id}/completions \
  -H "Authorization: Bearer {api_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "session_uuid",
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

### 7.3 进行RAG检索
```bash
curl -X POST http://localhost:9380/api/v1/retrieval \
  -H "Authorization: Bearer {api_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_ids": ["kb_id"],
    "query": "搜索内容",
    "top_n": 6
  }'
```

## 8. 环境配置

### 8.1 关键环境变量
| 变量 | 说明 | 默认值 |
|-----|------|-------|
| HOST_IP | API服务监听IP | 0.0.0.0 |
| HOST_PORT | API服务监听端口 | 9380 |
| MAX_CONTENT_LENGTH | 最大请求体大小 | 1GB |
| RAGFLOW_DEBUGPY_LISTEN | 调试端口 | 0 (关闭) |
| USE_DOCLING | 使用Docling解析 | false |
| USE_MINERU | 使用MinerU解析 | false |

### 8.2 配置文件
- **主配置**: `/conf/service_conf.yaml`
- **模板**: `/conf/service_conf.yaml.template`
- **初始化**: 启动时自动从模板生成

## 9. 数据库与存储

### 9.1 初始化流程
1. 初始化数据库表 (`init_web_db()`)
2. 初始化默认数据 (`init_web_data()`)
3. 初始化运行时配置 (`RuntimeConfig.init_env()`)

### 9.2 连接管理
- 自动释放数据库连接 (`@app.teardown_request`)
- 支持分布式锁 (`RedisDistributedLock`)
- 后台进程定期更新文档解析进度

## 10. 错误处理与响应

### 10.1 响应格式
所有API响应均遵循统一格式：
```json
{
  "code": 0,
  "message": "Success",
  "data": {}
}
```

### 10.2 错误代码
- 0: 成功
- RetCode.DATA_ERROR: 数据错误
- RetCode.AUTHENTICATION_ERROR: 认证错误
- RetCode.OPERATING_ERROR: 操作错误

### 10.3 全局异常处理
- 定义在 `/api/apps/__init__.py`
- 装饰器: `@app.errorhandler(Exception)(server_error_response)`

## 11. 扩展与定制

### 11.1 插件系统
- **插件管理器**: `GlobalPluginManager`
- **加载时机**: 应用启动时 (`GlobalPluginManager.load_plugins()`)
- **配置路由**: `/v1/plugin`

### 11.2 集成支持
- **Dify**: RAG检索集成
- **Langfuse**: LLM可观测性
- **MCP**: 协议支持
