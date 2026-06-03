# multiDal

多模态 RAG 系统,面向企业文档智能。摄入 PDF(报告、扫描件、财报、设计稿),提取文本与视觉内容,构建跨模态向量数据库,通过 LLM 实现图文混合问答,支持多知识库隔离与会话记忆。

## 架构一览

```
用户 → FastAPI → Kafka → Worker → MinerU(解析)
                                    ↓
                          TextEmbedder(BGE) + ImageEmbedder(Jina CLIP, 双路)
                                    ↓
                                Milvus(双 Collection)
                                   
用户 → FastAPI → KBRouterAgent(Qwen3-0.6B) → QueryRewriterAgent(Qwen3-0.6B)
                       ↓
              双路召回(文本 + 图像) → BCE Reranker → QueryAgent(MiniMax-M2.7) → SSE
```

| 组件 | 技术 | 用途 |
|------|------|------|
| PDF 解析 | MinerU (magic-pdf, vlm) | 云端 API → Markdown + 图片 + 表格 + LaTeX |
| 文本向量 | BGE large zh v1.5 (1024-d) | 语义匹配,Moark API |
| 图片向量 | Jina CLIP v2 (1024-d) | 像素 + 描述双路,Moark API |
| 图片描述(可选) | Qwen2.5-VL-7B-Instruct | 中文 caption,Moark API |
| 小模型(路由/改写/命名) | Qwen3-0.6B | 关闭 thinking,Moark API |
| 精排 | BCE reranker base v1 | Cross-encoder,Moark sentence-similarity API |
| 向量库 | Milvus | 双 Collection(`{kb_id}_text` + `{kb_id}_image`) |
| 消息队列 | Kafka | 异步解耦上传与处理 |
| 状态存储 | MySQL | `parse_tasks` / `knowledge_bases` / `agent_sessions` / `agent_messages` |
| Agent | openai-agents SDK | QA + 工具调用 + 会话记忆 |
| LLM | MiniMax-M2.7 | `https://api.minimaxi.com/v1` |
| 前端 | Vue 3 + Vite | Markdown / LaTeX / Mermaid / SSE 流式 / 会话管理 |
| 部署 | Docker Compose | 一键起 8 个服务 |

完整架构与设计决策见 [DESIGN.md](./DESIGN.md),项目难点与评测方案见 [EVAL.md](./EVAL.md),面经叙述见 [INTERVIEW.md](./INTERVIEW.md)。

## 快速开始

### 前置条件

- Docker & Docker Compose
- 8GB+ 内存(Milvus 要求)

### 1. 配置

编辑 `configs/settings.yaml`,填入你的 API 密钥:

```yaml
mineru:
  api_token: "your-mineru-token"
text_embedding:
  api_key: "your-moark-key"
image_embedding:
  api_key: "your-moark-key"
reranker:
  api_key: "your-moark-key"
vl_caption:
  api_key: "your-moark-key"
llm:
  api_key: "your-minimaxi-key"
  base_url: "https://api.minimaxi.com/v1"
  model: "MiniMax-M2.7"
small_llm:
  model: "Qwen3-0.6B"
```

### 2. 一键启动

```bash
docker-compose up -d
```

服务列表:`api`(8000)/ `worker` / `kafka`(9092)/ `zookeeper` / `milvus`(19530)/ `minio` / `etcd` / `mysql`(3306)/ `attu`(3000)

### 3. 打开前端

Docker 部署时 API 已内置前端 SPA:

```
http://localhost:8000
```

### 4. 使用流程

1. **新建知识库** → "+ 新建知识库"
2. **上传 PDF** → "文档上传"页,选 KB,拖拽上传
3. **等待处理** → "任务监控"页看进度(`pending → processing → completed / failed / exhausted`)
4. **智能问答** → "智能问答"页,选 KB,输入问题(SSE 流式输出)

## API 概览

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/ingest` | 上传 PDF(form-data: `file` + `kb_id`) |
| `GET` | `/api/ingest/{task_id}` | 查询处理进度 |
| `POST` | `/api/query` | 问答(非流式) |
| `POST` | `/api/query/stream` | 问答(SSE) |
| `POST` | `/api/kb/create` | 创建知识库 |
| `GET` | `/api/kb/list` | 列出知识库 |
| `DELETE` | `/api/kb/{kb_id}` | 删除知识库 |
| `GET` | `/api/kb/{kb_id}/docs` | 列 KB 文档 |
| `GET` | `/api/docs/{task_id}` | 查看完整文档(markdown) |
| `DELETE` | `/api/docs/{task_id}` | 删除文档 |
| `GET` / `POST` / `GET` / `PATCH` / `DELETE` | `/api/sessions[/{id}[/messages]]` | 会话 CRUD + 消息历史 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/raw/{task_id}/images/...` | 解析出的图片(StaticFiles) |

完整 OpenAPI:`http://localhost:8000/docs`

## 手动部署(开发)

```bash
python -m venv .venv && source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate on Windows

pip install -r requirements.txt

# 启动基础服务
docker-compose up -d etcd minio milvus zookeeper kafka mysql

# 启动 API
uvicorn src.multidal.api.app:app --reload --port 8000
# 或:python main.py

# 另开终端,启动 Worker
python -m src.multidal.queue.consumer
```

## 测试

```bash
pytest
pytest --cov=src/multidal --cov-report=term-missing
python main_demo.py            # TestClient 进程内集成测试
python main_demo.py --live     # 对运行中的服务做实时测试
```

## 项目结构

```
multiDal/
├── src/multidal/
│   ├── api/          # FastAPI 路由(ingest/query/kb/doc/sessions/health)
│   ├── agents/       # QueryAgent(主 LLM) + KBRouterAgent/QueryRewriterAgent(小模型)
│   │                 # + MySQLSession + function_tool(search_kb / get_doc_info)
│   ├── pipeline/     # Stage 抽象 + Orchestrator 驱动器
│   ├── parser/       # MinerU PDF 解析 + chart_detector
│   ├── embedder/     # Text(BGE) + Image(Jina CLIP, 双路) + VLCaptioner(可选)
│   ├── store/        # MilvusStore + MultiPathRetriever + Reranker
│   ├── kb/           # KBManager + IntentRouter/QueryRewriter(代理封装别名)
│   ├── llm/          # 小模型调用封装(Qwen3-0.6B)
│   ├── queue/        # Kafka Producer/Consumer
│   ├── db/           # MySQL 模型 + repository
│   ├── schema/       # Pydantic 数据模型(跨阶段契约)
│   ├── config/       # Pydantic Settings
│   └── utils/        # 图片预处理、日志
├── configs/settings.yaml
├── frontend/         # Vue 3 前端 SPA
├── scripts/          # CLI(ingest.py / query.py)
├── tests/
├── main.py           # uvicorn 启动包装
├── main_demo.py      # 集成测试(TestClient + --live)
├── docker-compose.yml
└── Dockerfile
```

## 核心模块说明

### 小模型 Agent 体系

路由、改写、命名都用 **Qwen3-0.6B**(Moark API),通过 openai-agents SDK 的 `Agent` 接口调用,**`enable_thinking=False`**:

| Agent | 任务 | 工具 |
|-------|------|------|
| `KBRouterAgent` | 用户问题 → JSON 数组 `["kb_id1", ...]` | `search_knowledge_base` / `get_doc_info` |
| `QueryRewriterAgent` | 原问题 → 2-3 个子查询(行分隔) | 无 |
| `generate_session_name` | 首问 + 答 → 3-10 字中文名 | 无 |

**降级原则**:任何小模型调用失败,上层捕获并退到"全量 KB / 原问题 / 问题前 15 字",**永远不让路由/改写错成为用户拿不到答案的原因**。

### Agent 函数工具

| 工具 | 作用 |
|------|------|
| `search_knowledge_base(query, kb_ids, top_k)` | BGE 嵌入 + 跨 KB Milvus 检索 + 去重,返回 top_k |
| `get_doc_info(doc_id)` | 查 MySQL,返回文件名 / 页数 / 状态 / kb_id |

两个工具都挂到 `KBRouterAgent`,让小模型在"是否真要查这个 KB"时能看到实际数据。

### 会话持久化

MySQL `agent_sessions` + `agent_messages` 表:

- `message_data` 存 LONGTEXT JSON(role / content / tool_calls)
- `sources` 单独存 JSON 数组,**按 message 关联**,前端按消息展示"参考了哪些文档"
- 首问答完 → `generate_session_name` 自动起名

### 双路召回与精排

文本 + 图片来自**不同向量空间**(BGE vs Jina CLIP),不能直接按召回分合并排序。处理:

1. **召回**:两条路径独立,合并去重
2. **精排**:文本调 BCE rerank API 重打分,图片保留 CLIP 分(已在正确空间)
3. **统一排序**:全部候选放一起按精排分降序,top-5 给 LLM

详见 [docs/debug_image_retrieval.md](./docs/debug_image_retrieval.md)(一次"图片为什么搜不到"的复盘)。
