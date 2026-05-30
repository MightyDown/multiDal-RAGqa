# multiDal

多模态 RAG 系统，面向企业文档智能。摄入 PDF（报告、扫描件、财报、设计稿），提取文本与视觉内容，构建跨模态向量数据库，通过 LLM 实现图文混合问答。

## 架构

```
用户 → FastAPI → Kafka → Worker → MinerU(解析) → Embedder → Milvus(存储)
                                    │
用户 → FastAPI ──→ 双路召回 + Rerank ──→ LLM 生成 ──→ SSE 流式回答
```

| 组件 | 技术 | 用途 |
|---|---|---|
| PDF 解析 | MinerU (magic-pdf) | 云端 API → Markdown + 图片 + 表格 + LaTeX |
| 文本向量 | BGE large zh v1.5 (1024-d) | 语义匹配，Moark API |
| 图片向量 | Jina CLIP v2 (1024-d) | 图文跨模态检索，Moark API |
| 精排 | BCE reranker base v1 | Cross-encoder 重排序，Moark API |
| 向量库 | Milvus | 双 Collection（text + image） |
| 消息队列 | Kafka | 异步解耦上传与处理 |
| 状态存储 | SQLite | 任务状态、知识库元数据、会话记忆 |
| Agent | openai-agents SDK | RAG 问答 + 会话记忆 |
| LLM | qwen-plus (阿里通义) | DashScope compatible 接口 |
| 前端 | Vue 3 + Vite | Markdown 渲染、LaTeX 公式、SSE 流式、会话管理 |
| 部署 | Docker Compose | 一键启动全部服务 |

完整架构详见 [DESIGN.md](./DESIGN.md)。

## 快速开始

### 前置条件

- Docker & Docker Compose
- 8GB+ 内存（Milvus 要求）

### 1. 配置

编辑 `configs/settings.yaml`，填入你的 API 密钥和模型端点：

```yaml
mineru:
  api_token: "your-mineru-token"
text_embedding:
  api_key: "your-moark-key"
image_embedding:
  api_key: "your-moark-key"
reranker:
  api_key: "your-moark-key"
llm:
  api_key: "your-dashscope-key"
```

### 2. 一键启动

```bash
docker-compose up -d
```

服务列表：api (8000)、worker、kafka (9092)、zookeeper (2181)、milvus (19530)、minio、etcd

### 3. 打开前端

**Docker 部署**（API 已内置前端 SPA）：

```
http://localhost:8000
```

**本地开发**（前端独立启动）：

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173，API 请求自动代理到 :8000
```

### 4. 开始使用

1. **新建知识库** → 点击"+ 新建知识库"
2. **上传 PDF** → 切换到"文档上传"页，选择知识库，拖拽 PDF 上传
3. **等待处理** → "任务监控"页查看进度（pending → processing → completed）
4. **智能问答** → 切换到"智能问答"页，选择知识库，输入问题

## API 概览

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/ingest` | 上传 PDF（form-data: file + kb_id） |
| `GET` | `/api/ingest/{task_id}` | 查询处理进度 |
| `POST` | `/api/query` | 问答（非流式） |
| `POST` | `/api/query/stream` | 问答（SSE 流式） |
| `POST` | `/api/kb/create` | 创建知识库 |
| `GET` | `/api/kb/list` | 列出知识库 |
| `DELETE` | `/api/kb/{kb_id}` | 删除知识库 |
| `GET` | `/api/kb/{kb_id}/docs` | 列出知识库文档 |
| `GET` | `/api/docs/{task_id}` | 查看文档完整内容 |
| `DELETE` | `/api/docs/{task_id}` | 删除文档 |
| `GET` | `/api/sessions` | 列出会话历史 |
| `DELETE` | `/api/sessions/{id}` | 删除会话 |
| `GET` | `/api/health` | 健康检查 |

完整交互式文档：`http://localhost:8000/docs`

## 手动部署（开发）

```bash
# 创建虚拟环境
python -m venv .venv && .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 启动基础服务（Milvus + Kafka）
docker-compose up -d etcd minio milvus zookeeper kafka

# 启动 API
uvicorn src.multidal.api.app:app --reload --port 8000

# 另开终端，启动前端（可选，Docker 部署时 API 已内置前端）
cd frontend && npm install && npm run dev

# 另开终端，启动 Worker
python -m src.multidal.queue.consumer
```

## 测试

```bash
pytest
pytest --cov=src/multidal --cov-report=term-missing
```

## 项目结构

```
multiDal/
├── src/multidal/
│   ├── api/          # FastAPI 路由
│   ├── agents/       # QueryAgent + 会话管理
│   ├── pipeline/     # Stage 抽象 + 编排
│   ├── parser/       # MinerU PDF 解析
│   ├── embedder/     # 文本 + 图片向量化
│   ├── store/        # Milvus + 多路召回 + Rerank
│   ├── kb/           # 知识库管理 + 意图路由 + Query改写
│   ├── queue/        # Kafka Producer/Consumer
│   ├── db/           # SQLite 模型
│   ├── schema/       # 数据模型
│   ├── config/       # Pydantic Settings
│   └── utils/        # 图片预处理、日志
├── configs/          # YAML 配置（pipeline / kafka / 模型 API / LLM）
├── frontend/         # Vue 3 前端 SPA
├── scripts/          # CLI 工具
├── tests/            # 测试
├── docker-compose.yml
└── requirements.txt
```