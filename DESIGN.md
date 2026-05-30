# multiDal — 多模态 RAG 系统架构设计

## 项目背景

企业数字化转型中，核心知识资产不再局限于纯文本。PDF 报告、扫描件、财务图表、设计图稿等多模态文档需要深度语义理解。传统文本 RAG 和 OCR 技术存在局限性：无法把握图表数据趋势、图像对象关系及复杂版面结构，缺乏跨模态检索能力。

## 四阶段流水线

```
PDF 上传 → Parser(MinerU) → Embedder(BGE + Jina CLIP) → Store(Milvus) → RAG(双路召回+Rerank+Agent)
```

| 阶段 | 输入 | 处理 | 输出 |
|---|---|---|---|
| Parser | PDF 文件 | MinerU 云端 API → Markdown + 图片 + 表格 + LaTeX + 版面 JSON | ParsedDocument |
| Embedder | ParsedDocument | 文本: BGE large zh v1.5 (1024d, Moark API) / 图片: Jina CLIP v2 (1024d, Moark API) | EmbeddedDocument |
| Store | EmbeddedDocument | 写入 Milvus 双 Collection（text 1024d + image 1024d） | 向量索引 |
| RAG | 用户问题 | KB 路由 → Query 改写 → 双路召回 → BCE-reranker 精排 → LLM 生成 | 带引用的答案 |

## 项目结构

```
multiDal/
├── src/multidal/
│   ├── config/          # Pydantic Settings（configs/settings.yaml）
│   ├── schema/          # Pydantic 数据模型（跨阶段数据契约 + JSON 验证）
│   ├── pipeline/        # Stage 抽象基类 + Orchestrator
│   ├── queue/           # Kafka producer / consumer
│   ├── parser/          # MinerU 云端 API 包装器 + 图表检测
│   ├── embedder/        # 文本向量化(BGE) + 图片向量化(Jina CLIP) + 模型注册
│   ├── store/           # MilvusStore + MultiPathRetriever + Reranker
│   ├── db/              # SQLite 模型(parse_tasks, knowledge_bases) + repository
│   ├── agents/          # BaseAgent → QueryAgent / IngestAgent / 会话管理
│   ├── kb/              # KBManager + IntentRouter + QueryRewriter
│   ├── api/             # FastAPI (ingest / status / query / kb / doc / sessions / health)
│   └── utils/           # 图片预处理、日志
├── configs/
│   └── settings.yaml    # 所有配置（pipeline / kafka / milvus / 模型 API / LLM）
├── frontend/           # Vue 3 + Vite 前端 SPA
├── tests/
├── scripts/
├── docker-compose.yml
└── Dockerfile
```

## 核心技术选型

| 组件 | 选择 | 说明 |
|---|---|---|
| 文档解析 | MinerU (magic-pdf) | 云端 API `https://mineru.net`，JWT 认证 |
| 文本 Embedding | BGE large zh v1.5 | 1024 维，Moark API (`/embeddings`) |
| 图片 Embedding | Jina CLIP v2 | 1024 维，Moark API (`/embeddings`)；每张图片产生像素向量+描述向量共 2 条 |
| Rerank | BCE reranker base v1 | Cross-encoder 精排，Moark API (`/sentence-similarity`) |
| 向量数据库 | Milvus | Docker 容器，端口 19530 |
| 消息队列 | Kafka | Docker Compose 管理 |
| 状态存储 | SQLite | parse_tasks 表 + knowledge_bases 表 + sessions.db |
| Agent 框架 | openai-agents SDK | 工具调用 + 会话记忆 |
| LLM | qwen-plus (阿里通义) | DashScope compatible OpenAI 接口 |

## 检索链路

```
用户问题
  → KB 路由（手动指定 / IntentRouter LLM 自动识别）
  → Query 改写（原始问题 → 最多 3 个多角度搜索词，LLM 生成）
  → 双路并行召回：
     A: BGE Embedding (1024d) → Milvus {kb}_text collection（语义检索）
     B: Jina CLIP text encoder (1024d) → Milvus {kb}_image collection（跨模态检索）
  → 合并去重（按 kb_id:chunk_id 唯一化，保留最高分）
  → BCE-reranker 精排（top-5，Moark sentence-similarity API）
  → 上下文组装 → LLM 生成答案（SSE 流式）
```

**注意：代码中实际只有 A+B 两路召回，不存在 BM25 或"描述反查"路径。**

## 多知识库

- 每个 KB 对应 Milvus 中一对独立的 Collection（`{kb_id}_text` + `{kb_id}_image`，物理隔离）
- SQLite `knowledge_bases` 表维护 KB 元数据
- `parse_tasks.kb_id` 关联文档与知识库
- 查询时手动指定 `kb_ids`，或 `auto_route=true` 由 IntentRouter LLM 自动路由

## LLM 策略

不要求多模态模型。图片内容以 MinerU 生成的**文字描述 caption** 传入 LLM，任何文本 LLM 均可使用。

## 解析可靠性

SQLite `parse_tasks` 表实时跟踪状态（pending → processing → completed / failed / exhausted）。失败自动重传 Kafka（退避 30s → 60s → 90s），超过 `max_retries` 标记 exhausted。

## 配置

单一配置文件 `configs/settings.yaml`，通过 Pydantic Settings 加载，环境变量可覆盖：

| Section | 用途 |
|---------|------|
| `pipeline` | top_k_recall (default 10), top_k_final (default 5), max_retries (default 3) |
| `kafka` | bootstrap_servers, topic |
| `milvus` | host, port |
| `db` | path (default data/multidal.db) |
| `mineru` | api_base, api_token, model_version |
| `text_embedding` | api_base, api_key, model (bge-large-zh-v1.5), dim (1024) |
| `image_embedding` | api_base, api_key, model (jina-clip-v2), dim (1024), device |
| `reranker` | api_base, api_key, model (bce-reranker-base_v1) |
| `llm` | api_key, base_url, model (qwen-plus) |
| `log` | level |

## 关键细节

### 入库流程（Consumer Worker）

```
[MinerUParser] → [TextEmbedder] → [ImageEmbedder?可选] → [MilvusStore]
```

`ImageEmbedder` 默认插入 TextEmbedder 之后，通过 `validate()` 检测可用性（需 CUDA + Jina CLIP 模型）决定是否启用。

### ImageEmbedder 双路编码

每张图片产生 **2 个 chunk** 存入 `{kb}_image` collection：

| chunk_id | content | modality | 向量来源 |
|----------|---------|----------|---------|
| `img_abc123` | "收益趋势折线图..." | image | Jina CLIP image encoder（像素→768d） |
| `img_abc123_desc` | "收益趋势折线图..." | image | Jina CLIP text encoder（描述文字→768d） |

`_load_image_uri`：读取磁盘图片 → 缩放到 max 168px → JPEG 质量 30 → base64 data URI → CLIP image encoder

### 查询时向量分发

| 查询内容 | 使用模型 | 目标 Collection | 维度 |
|----------|---------|-----------------|------|
| 文本 query | BGE Embedding | `{kb}_text` | 1024d |
| 文本 query（搜图库） | Jina CLIP text enc | `{kb}_image` | 1024d |

Query 文本同时走 A 路（BGE→text collection）和 B 路（CLIP→image collection），结果合并去重。

### 会话管理

openai-agents SDK 的 `SQLiteSession` 持久化到 `data/sessions.db`（独立于主数据库）：
- `agent_sessions`: session_id, session_name, created_at, updated_at
- `agent_messages`: id, session_id, message_data, created_at（CASCADE 删除）

首个问题回答完成后，LLM 自动生成会话名称（3-10 汉字），存入 `agent_sessions.session_name`。