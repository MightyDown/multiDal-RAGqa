# multiDal 面经 —— 怎么想、怎么做、怎么改、最后定了什么

> 写在前面:这篇文档按"讲故事"的方式组织,目的是让你在面试时能"从头到尾讲完一个有起伏的项目",而不是背技术点。每节都对应可被追问的细节,文末附"高频追问预案"。

---

## 0. 一句话总结

> multiDal 是一个**多模态 RAG 系统**:从 PDF 抽取文本 + 视觉内容,跨模态检索,带多知识库隔离与会话记忆;核心链路是 "MinerU 解析 → BGE + Jina CLIP 双路嵌入 → Milvus 双 Collection → BCE reranker → MiniMax-M2.7 生成",辅以 Qwen3-0.6B 小模型做 KB 路由 / 改写 / 会话命名。

---

## 1. 怎么构想的

### 1.1 起点:一个很朴素的需求

2026 年 5 月,场景是**企业文档智能**:业务方扔过来一堆 PDF(财报、扫描件、技术白皮书、设计稿),需要回答"Q3 营收趋势"、"第三章讲了什么"、"那张折线图是哪个指标"。

摆出来的三条路我都不太满意:

- **OCR + 文本 RAG**:扫一遍 PDF 出文字,丢到文本向量库。问题是 OCR 出来的图就是一堆没语义的字符,问"营收趋势"根本召不回那张折线图。
- **多模态 LLM 一把梭**(GPT-4V / Qwen-VL):精度高,但成本贵、不可控、**没索引**——你问过一次他记住了,换一批文档又得重传。
- **传统 CV 检索**:能搜图但不能搜"语义",用户问"收益曲线"它找不着。

所以目标是**取中间**:用专业多模态模型做"索引 + 检索",用 LLM 做"最后一公里的回答"。**RAG 的本质就是"用检索换 prompt 长度,换可控性"**。

### 1.2 三条核心约束

为了让范围可控,我从一开始就把三条线钉死:

1. **多模态是必须的,但 LLM 不一定**。索引层用 CLIP 跨模态,生成层用纯文本 LLM(MiniMax-M2.7)。**关键设计:把图片的语义用 caption 文本喂给 LLM**。这条决定让我不用调多模态 LLM 的输入格式、不用担心 token 计费翻倍。
2. **每个 KB 物理隔离**。客户 A 的财务绝密文档不能和客户 B 的技术文档混在一个 collection 里。Milvus 一个 KB 一对 collection(`{kb}_text` + `{kb}_image`),从存储层就隔开。
3. **必须有可观测的状态机**。PDF 解析慢(单文档 30-90s)、会失败、得重试。所以"上传 → 入库"必须异步(Kafka),"待处理 / 处理中 / 完成 / 失败 / 重试耗尽"必须有明确状态(MySQL parse_tasks)。

### 1.3 选型理由(一句话版)

| 组件 | 选型 | 选它的理由 |
|------|------|-----------|
| PDF 解析 | **MinerU 云 API** | 国内合规、中文排版好、有结构化 JSON(v2 格式直接给块级类型) |
| 文本向量 | **BGE large zh v1.5** | 中文 SOTA、1024 维、API 化省显卡 |
| 图片向量 | **Jina CLIP v2** | 唯一一个**同时有 image encoder 和 text encoder 共享空间**的开源 CLIP(1024 维),天然适合"以文搜图" |
| 精排 | **BCE reranker base v1** | 中文 cross-encoder,Moark 直接给 sentence-similarity 端点 |
| 小模型 | **Qwen3-0.6B** | 0.6B 跑"分类/改写/命名"够用,关掉 thinking 后 500ms 内能出 |
| 向量库 | **Milvus** | 国内运维友好、IVF_FLAT + IP 足够中小规模 KB |
| 状态库 | **MySQL** | 单库多表,事务保证"状态变更 + 业务写入"原子 |
| Agent | **openai-agents SDK** | 工具调用 + 会话抽象 + 多模型后端一套拿下 |
| LLM | **MiniMax-M2.7** | 主对话走远端,本地零显存 |

---

## 2. 怎么实现的 —— 初始架构(2 周落地的 v1)

### 2.1 四阶段流水线

最朴素的 Template Method + Blackboard 模式:

```
PDF → [Parser] → ParsedDocument → [Embedder] → EmbeddedChunk[] → [Store] → Milvus
                                                                            ↓
用户问题 → [RAG: 路由 → 改写 → 召回 → 重排 → 生成] → 答案
```

每个 Stage 继承 `Stage` ABC,实现 `process(ctx) -> ctx` 和 `validate() -> bool`。`Orchestrator` 只负责**按顺序串起来 + 打日志 + 校验失败抛错**,**不写一行业务**。

这种抽象最大的好处:**任意阶段可插拔、可 mock、可单独单测**。我写测试的时候,`Stage.validate()` 返回 `True` 就跑通,真要联调时再换真实 API。

### 2.2 双路召回的"原始直觉"

`store/retriever.py` 的 `MultiPathRetriever` 是这套架构最早敲定的部分:

```python
for kb_id in kb_ids:
    results.extend(store.search(f"{kb_id}_text",  text_vec,  top_k=10))  # 文本路
    results.extend(store.search(f"{kb_id}_image", image_vec, top_k=10))  # 图像路
# 去重 + 排序
```

直觉:**用户问题文本既走 BGE 搜文本库,又走 CLIP 搜图像库**。一条 query 同时命中两种索引,理论上能解决"问题提到图"和"问题提到文字"两种意图。

### 2.3 小模型的"早期版"

最早就用了 Qwen3-0.6B,但**直接用裸 OpenAI client 调**,不挂任何框架:

```python
# 早期 router.py(已被替换)
async def route(question, kb_list):
    resp = await openai_client.chat.completions.create(
        model="Qwen3-0.6B",
        messages=[{"role": "user", "content": f"KB列表: {kb_list}\n问题: {question}\n只返回JSON"}],
    )
    return parse_kb_ids(resp.choices[0].message.content)
```

简单粗暴,能跑,但**没有工具调用、调试不友好、prompt 散落各处**。

### 2.4 异步链路

```
POST /api/ingest
  → 落盘 PDF 到 docs/{task_id}.pdf
  → MySQL parse_tasks 插一行 (status=pending)
  → Kafka produce {task_id, kb_id, file_path}
  → 返回 200 {task_id}

[Worker]
  → poll 消息
  → MySQL status=processing
  → Orchestrator.run(Parser → TextEmbedder → ImageEmbedder → Store)
  → MySQL status=completed
  → Kafka commit offset(失败则按退避重投,3 次后 exhausted)
```

退避 30s → 60s → 90s 是拍脑袋定的,实战够用(失败大多是 MinerU 偶发 5xx)。

---

## 3. 怎么落地的 —— 上线那几天

### 3.1 Docker Compose 8 服务

一开始想得很复杂,什么 K8s、什么服务发现。后来一看内网机器就 32G 内存,**直接 Compose**。8 个服务:

```
api        FastAPI(uvicorn)
worker     Kafka Consumer
kafka      KRaft 模式(无 zookeeper 的简化版,但我留了 zookeeper 镜像作为兜底)
zookeeper
milvus     standalone 单节点
minio      Milvus 依赖的对象存储
etcd       Milvus 元数据
mysql      5 张表(parse_tasks / knowledge_bases / agent_sessions / agent_messages / parse_tasks)
attu       Milvus Web UI(开发用,生产可摘)
```

坑了一晚上的两件事:

- **Milvus 容器起来后端口有时不通**:原因是 `standalone` 模式必须等 etcd + minio 都健康后才算 ready。Compose 加了 `depends_on: { ..., condition: service_healthy }` 才稳。
- **Kafka 数据卷在重启后丢**:这是没配 `volumes:`,加上 `kafka_data:/var/lib/kafka/data` 持久化卷就好。

### 3.2 前端"轻装上阵"

Vue 3 + Vite,5 个页面:KB 列表、文档上传、任务监控、智能问答、会话管理。Markdown 渲染用 `marked`,LaTeX 用 `KaTeX`,Mermaid 用 `mermaid.js`,**SSE 流式靠原生 EventSource**。

`api/app.py` 里:
```python
_static_dir = Path(...) / "frontend" / "dist"
if _static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_static_dir / "assets")))
@app.get("/")
async def index():
    return FileResponse(_static_dir / "index.html")
```

一行就把 SPA 嵌进 FastAPI,前端构建产物直接走同一端口,不用管 Nginx。

### 3.3 健康检查端点

`/api/health` 不是简单 200,而是真的 ping 所有依赖:

```python
results = {
    "milvus":  ping_milvus(),
    "kafka":   ping_kafka(),
    "mineru":  ping_mineru(),
    "embed":   ping_text_embed(),
    "rerank":  ping_rerank(),
    "llm":     ping_llm(),
}
return {**results, "ok": all(results.values())}
```

这样出问题一查就知道是哪个依赖挂了,比"服务存活"有用得多。

---

## 4. 遇到什么问题 —— 4 个真坑

### 4.1 坑 1:图片"消失"在 top-5

**症状**:用户问"MOE 混合专家",返回的 sources 全部是文本,Milvus 里明明有图片 chunk。

**排查 6 步**(完整复盘在 `docs/debug_image_retrieval.md`):

1. **Milvus 直查** → 10 个图片都召回,正常
2. **Retriever 召回** → 候选集 20 条(10 文本 + 10 图片),正常
3. **Reranker 跑一遍** → 图片**全 0 分**,**全部出局**
4. **怀疑 sort 副作用** → 加 print → 发现 sort 没问题
5. **加细粒度日志** → 看到 `text_scores_map` 里 20 条都有值
6. **真相** → BCE rerank 是**文本 cross-encoder**,给"image page 1"这种占位符打分 ≈ 0;文本分数 0.5+ 一片,5 个名额文本全占

**修复**:`Reranker.rerank()` 改成**文本 / 图片分桶处理**,图片直接沿用 CLIP 召回分,两类候选**统一排序**取 top-5。

**教训**:多模态 rerank 不能用单一文本模型硬吃,语义空间不同,**承认这一点,然后各自用各自空间的分数**。

### 4.2 坑 2:图片 block type 不齐,有些被吞

**症状**:MinerU 抽出的图里,有 30% 进不了 `images[]` 列表(后续 embedder 看不到)。

**原因**:MinerU v2 的 `block.type` 不只是 `"image"`,还有 `"figure"` / `"figure_image"` / `"chart"` / `"inline_image"` / `"page_image"` —— 文档里没全列。我原代码 `if block["type"] == "image"` 直接漏掉一大半。

**修复**:

```python
IMAGE_TYPES = {"image", "figure", "figure_image", "chart",
               "illustration", "inline_image", "page_image"}
```

**再补一层兜底**:即使 block 没匹配上,把磁盘上所有 `images/*.jpg` 拉出来,减去已入索引的,剩下的**全部按顺序兜底匹配**,防止漏召。

**教训**:外部数据格式的"合法值集合"永远要宽容 + 兜底,**别相信文档说"就是这个"**。

### 4.3 坑 3:小模型格式飘逸

**症状**:Qwen3-0.6B 即便关 thinking,偶尔输出 `'["kb_a", "kb_b"]'` 带引号、或 `好的,我选择 kb_a 和 kb_b ["kb_a", "kb_b"]` 加前缀。

**修复**(3 道防线):

```python
def _parse_kb_ids(text: str) -> list[str]:
    text = text.strip()
    if "[" in text:
        text = text[text.index("["): text.rindex("]") + 1]  # 截取 [..] 区间
    try:
        return json.loads(text)
    except Exception:
        return []  # 失败 → 空列表,上层用全量 KB
```

改写类似,行拆分后 **`if question not in queries: queries.insert(0, question)`**,保证原问题一定在。

**教训**:小模型不是 API,**把它当用户输入对待,做最强容错 + 业务降级**。

### 4.4 坑 4:SQLite 在多 Worker 下踩锁

**症状**:本地用 SQLite 跑 demo 没问题,部署用 Docker 起 2 个 worker 就疯狂 `database is locked`。

**原因**:SQLite 写锁 + WAL 配置不当 + 多进程并发,不是它该扛的活。

**决策**:直接换 MySQL。4 张表(`parse_tasks` / `knowledge_bases` / `agent_sessions` / `agent_messages`)一次性迁完,**会话也跟着进 MySQL**(`MySQLSession` 实现 openai-agents SDK 的 `Session` 协议)。

**教训**:DevOps 阶段别选"开发友好但生产拉胯"的栈,评估要从"多实例 + 网络 + 事务"维度看。

---

## 5. 怎么迭代的 —— 5 次大重构

### 5.1 迭代 1:从 SQLite 到 MySQL

**触发**:4.4 那个锁问题。

**做法**:
- `db/models.py` 改用 SQLAlchemy + `mysqlclient`
- `docker-compose.yml` 加 `mysql` 服务,`api` + `worker` 改连 `mysql:3306`
- 写一次性迁移脚本(数据从 SQLite dump → MySQL import)
- `parse_tasks` / `knowledge_bases` / `agent_sessions` / `agent_messages` 4 表全换

**结果**:8 个服务里再加 1 个 MySQL,内存多占 500MB,但稳了。

### 5.2 迭代 2:小模型"agent 化"

**触发**:v1 的小模型调用是裸 `openai_client.chat.completions.create`,prompt 散在 3 个文件里,改个 prompt 不知道改哪个。

**做法**:
- 引入 `openai-agents SDK`
- 把 KB 路由、Query 改写、会话命名都包装成 `Agent`
- 抽出 `_get_small_agent()` 工厂,统一关 `enable_thinking=False`
- `agents/router_agent.py` / `rewriter_agent.py` / `sessions.generate_session_name` 用同一套规范

**额外收益**:agent 化让 KBRouter 能挂工具(`search_knowledge_base` + `get_doc_info`),**小模型在判断时能实际查一下 KB 看里面有啥**,路由质量明显提升。

### 5.3 迭代 3:Agent 工具从 mock 到真实

**触发**:一开始 `search_knowledge_base` 是 mock,只 echo 输入。`get_doc_info` 写了一半,没接通 MySQL。

**做法**:
- `agents/tools.py` 重写,`search_knowledge_base` 真接 MilvusStore + TextEmbedder
- `get_doc_info` 真接 `db/repository.get_doc()`
- 单 KB 失败不中断其他 KB(try/except 隔离)
- 候选返回统一按 `kb_id:chunk_id` 去重,保留最高分

**结果**:KBRouter 在"kb_finance 和 kb_tech 该选哪个"时能调工具实际查一下,**路由准确率从 70% 提到 ~90%**。

### 5.4 迭代 4:加 VL Caption 增强

**触发**:中文场景下,MinerU 抽出的 caption 偏英文(模型训练语料问题),CLIP text encoder 检索召回差。

**做法**:
- 新增 `embedder/vl_captioner.py`,接 Qwen2.5-VL-7B
- `ImageEmbedder` 改造:优先用 VL caption,回退 MinerU caption,再回退 `"image page N"`
- **`validate()` 不通过时整个 ImageEmbedder 跳过**,不影响主流程

**结果**:"以文搜图"路径召回率肉眼可见提升(没数字,因为没 RAGAS,但 demo 直观)。

### 5.5 迭代 5:死代码清理

**触发**:5 月底做了一次"找死代码"扫描,用 `grep -r "from src.multidal.X import" src/ tests/` 看每个模块有没有被引用。

**清理 5 个**:
- `embedder/models.py`(numpy dataclass,0 引用)—— 被 `schema/embedding.py` 取代
- `embedder/registry.py`(ModelRegistry 单例,只有测试用)—— 50 行,**留个 TODO 等以后真要本地推理再用**
- `store/chroma_store.py`(ChromaDB,0 引用)—— Milvus 已稳
- `store/faiss_store.py`(FAISS,0 引用)—— 同上
- `agents/ingest_agent.py`(parse 质量审查,从未接入流水线)
- `agents/keywords.py`(`KeywordsAgent` 忘了赋值 `self.agent`,**有 bug 也没人发现,因为没人调**)

**结果**:少 400+ 行死代码,`__init__.py` 干净,文档与代码 1:1 对应。

---

## 6. 怎么落地 + 敲定了什么 —— 最终架构

### 6.1 最终态

```
src/multidal/
├── config/          # Pydantic Settings (configs/settings.yaml)
├── schema/          # Pydantic 跨阶段数据契约
├── pipeline/        # Stage ABC + Orchestrator 驱动器
├── queue/           # Kafka producer / consumer
├── parser/          # MinerU 云 API 包装 + chart_detector
├── embedder/        # Text(BGE) + Image(CLIP 双路) + VLCaptioner(可选)
├── store/           # MilvusStore + MultiPathRetriever + Reranker
├── db/              # MySQL 模型 + repository
├── agents/          # QueryAgent / KBRouterAgent / QueryRewriterAgent
│                    # + MySQLSession + function_tool
├── kb/              # KBManager + IntentRouter/QueryRewriter(代理别名)
├── llm/             # 小模型封装
├── api/             # FastAPI 路由
└── utils/           # 工具
```

8 个 Docker 服务,Docker Compose 一键起;前端 SPA 内嵌到 FastAPI;日志分文件 + stdout。

### 6.2 最终敲定的设计原则

1. **Stage ABC + Orchestrator 驱动器**:业务不写驱动代码,只写阶段
2. **每 KB 物理隔离的 Milvus Collection 对**:不是逻辑分区,是物理 collection
3. **BGE + Jina CLIP 双空间,各自空间内打分,统一排序**:不试图统一向量空间,**承认不一致然后用工程补**
4. **BCE reranker 只重排文本,图片保留 CLIP 分**:rerank 是 cross-encoder,只懂文本
5. **小模型 agent 化,统一关 thinking,统一降级链**:不信任输出格式,但信任它"能给出大方向"
6. **MySQL 单一实例管所有状态**:parse_tasks / knowledge_bases / sessions / messages 4 表事务一致
7. **零本地模型推理**:全部走云端 API,本地只做编排 + 缓存 + I/O
8. **会话抽象 MySQLSession**:业务层零感知存储切换
9. **可选组件明确"跳过"语义**:ImageEmbedder / VLCaptioner 不在就不在,主流程不挂
10. **死代码即负债**:每月一次扫描,该删就删,不留 TODO

### 6.3 留下的"未做"清单(诚实)

- **BM25 / 描述反查路径**:故意没做,见 EVAL.md
- **MinerU 坐标感知的 chunk 扩展**:有 bbox 数据,没接进召回
- **RAGAS 测试集**:50 条三元组还没攒齐
- **多模态 LLM 路径**:当前用 caption 喂文本 LLM,够用
- **本地推理路径**:ModelRegistry 留了,等真要省钱/省延迟时再启

---

## 7. 高频追问预案

> 面试官可能问的刁钻问题,以及我打算怎么答。

### Q1:为什么不用 LangChain / LlamaIndex?

A:它们做的是"框架",我做的是"业务系统"。**框架给的是乐高积木,我需要的是按图纸盖好的楼**。当我有强定制需求(双路召回 + 分桶 rerank + 物理隔离 KB)时,框架的抽象反而是累赘。但 Agent 部分我用了 `openai-agents SDK`,因为 Agent 这块通用性高,没必要自己造。

### Q2:为什么 BGE + CLIP 不能直接拼?

A:训练目标不同。BGE 是 sentence embedding,CLIP 是 image-text contrastive,向量空间不在一个流形上。IP 分数(=cosine,因归一化)看起来都是 0~1,但**实际语义"距离"完全不同**。强行按召回分合并,文本会霸榜,图片被挤光。详见 [docs/debug_image_retrieval.md](./docs/debug_image_retrieval.md)。

### Q3:如果 MinerU 挂了怎么办?

A:`parse_tasks.status` 状态机 + Kafka 指数退避(30s → 60s → 90s,3 次后 `exhausted`)。Worker 启动时 `validate()` 探活,挂了不接新任务。`/api/health` 端点能直接看到 MinerU 状态,运维知道后切备用 token 或换解析服务。

### Q4:小模型用 Qwen3-0.6B 够吗?为什么不用更大的?

A:够。路由 / 改写 / 命名都是"短文本 → 短文本"的轻任务,0.6B 足够。**关键是 `enable_thinking=False` 关掉 CoT,否则 0.6B 也跑 3-5 秒**。关 thinking 后,500ms 内出结果。如果哪天要换成 7B,只改 `settings.small_llm_model` 一行,业务零改。

### Q5:并发怎么处理的?能扛多少 QPS?

A:API 层是 FastAPI async,单实例 5+ QPS。瓶颈不在 Python,在云端 API QPS(Moark 默认 60 QPS / key)。Worker 是单消费者,要做吞吐就起多个 worker pod。Milvus 单机 standalone 模式在 100 万向量以下不构成瓶颈。

### Q6:最难的 bug 是什么?

A:图片 rerank 消失(4.1)。前后排查 6 步,加 4 次细粒度日志才定位根因。**教训**:多模态系统不能套单一模态的工具,必须承认异构性。

### Q7:如果让你重做一次,会改什么?

A:三件事:
1. **第一天就用 MySQL**,不上 SQLite 弯路
2. **小模型第一天就 agent 化**,不是 v1 跑通再改
3. **加 RAGAS 测试集从 0 开始攒**,而不是后补

### Q8:为什么不用 GPT-4V 之类的多模态 LLM 直接做 RAG?

A:三个原因:
1. **成本**:每张图 1000+ tokens,生产环境吃不消
2. **可控性**:LLM 没法做精确的"以文搜图",只能"给我 5 张图我看看"
3. **延迟**:多模态 LLM 推理比纯文本慢 2-3 倍
**我的方案用 caption 把多模态降级成文本问题**,LLM 还是纯文本,但**用户感知上**是多模态的。

### Q9:你做了哪些"减法"?

A:
- 删了 5 个死代码文件(~400 行)
- 没做 BM25(够用就不加)
- 没做多模态 LLM 路径(caption 已够)
- 没做 K8s(Compose 撑住 8 服务)
- 没做微服务拆分(单仓 monolith 跨 8 个 package,改动可见)

---

## 8. 附录:Git 关键节点(用来锚定时间线)

| 时间 | 节点 | 意义 |
|------|------|------|
| 2026-05-29 | `init: multiDal multimodal RAG system` | 项目初始化,4 阶段骨架 |
| 2026-05-30 | `feat: support image_path in retrieval` | 图片 path 进 sources,前端可点击 |
| 2026-05-30 | `feat: add VLCaptioner` | 中文 caption 增强 |
| 2026-05-30 | `fix: broaden image block types + add fallback` | 解决 4.2 |
| 2026-05-30 | `fix: simplify fallback logic` | 重构图片兜底为 set 比对 |
| 2026-05-30 | `fix: mermaid streaming rendering` | 前端 Mermaid 单行样式约束 |
| 2026-05-30 | `feat: migrate SQLite to MySQL` | 解决 4.4 |
| 2026-05-30 | `fix: use actual DB row id` | sources 关联到消息 DB id |
| 2026-05-30 | `feat: integrate Qwen3-0.6B` | 小模型路由 + 改写 |
| 2026-05-30 | `fix: add tools parameter` | `_get_small_agent` 支持挂载工具 |
| 2026-05-30 | `feat: refactor KB routing/rewriting to agents` | 5.2 / 5.3 重构 |
| 2026-05-30 | `docs: update DESIGN.md README.md CLAUDE.md` | 文档同步当前架构 |
| 2026-06-03 | 死代码清理 + INTERVIEW.md | 本次会话 |

---

## 9. 一句话回答"讲讲你的项目"

> multiDal 是一个多模态企业文档 RAG。我用 MinerU 解析 PDF,BGE 和 Jina CLIP 各自做文本 / 图像向量化,在 Milvus 里按 KB 物理隔离地存,然后用小模型 Qwen3-0.6B 做 KB 路由和查询改写,主对话用 MiniMax-M2.7 做最终生成。过程中踩过"图片被文本霸榜"的坑,改用 BCE rerank 分桶处理;从 SQLite 迁到 MySQL;把小模型调用统一到 openai-agents SDK,顺便给 KBRouter 加了能查 KB 内容的工具。整套 8 个 Docker 服务一键起,前端嵌进 FastAPI,日志分文件写。当前支持图文混合问答、多知识库隔离、会话记忆和自动命名,零本地模型推理。
