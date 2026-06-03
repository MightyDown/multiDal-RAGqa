# multiDal — 项目难点与评测方案

> 与代码同步的难点分析 + 评测设计。配套 [DESIGN.md](./DESIGN.md) 与 [INTERVIEW.md](./INTERVIEW.md)。

## 一、项目难点

### 1.1 双 Embedding 模型的向量空间差异

```
文本路径: query → BGE Embedding (1024d) → Milvus {kb}_text
图片路径: query → Jina CLIP text encoder (1024d) → Milvus {kb}_image
```

BGE 与 Jina CLIP 训练目标不同,语义空间**不可比**:

- **召回分不可比**:路径 A 的 IP 分数和路径 B 的 IP 分数分属不同空间,直接合并排序毫无意义
- **图片不能用 sentence-similarity 精排**:BCE reranker 是文本 cross-encoder,给图片 caption("image page 1" 这种占位符)打分几乎为 0,会让图片被文本霸榜
- **正确做法**:文本走 BCE rerank 重打分,图片沿用 CLIP 召回分(已在正确空间),**两类候选统一排序**(详见 [docs/debug_image_retrieval.md](./docs/debug_image_retrieval.md))
- **ImageEmbedder 每图 2 条向量**:像素 + 描述共存于 `{kb}_image` collection,都参与检索

### 1.2 PDF 质量天花板

MinerU 云端 API 输出是下游所有阶段的输入,质量问题会逐级放大。

| 场景 | MinerU 表现 | 下游影响 |
|------|------------|---------|
| 扫描件(无文本层) | OCR 质量取决于图片清晰度 | 文本提取错误 → 向量质量崩塌 |
| 混合排版(中英混排、竖排) | 阅读顺序可能错乱 | chunk 语义断裂 |
| 复杂表格(合并单元格、嵌套) | HTML 表格结构可能不完整 | 表格查询失败 |
| 水印 / 印章 / 手写批注 | 可能混入正文 | 噪声污染向量库 |
| CAD / 设计图稿 | 可能不被识别 | 直接丢失 |
| 加密 PDF | 完全无法打开 | 任务直接失败 |
| MinerU 抽出 caption 偏英文 | 中文场景"以文搜图"召回差 | VLCaptioner 增强可缓解 |

**应对**:v2 格式优先 + v1 兜底,文本双 fallback(JSON 块 + markdown 段落),图片 block 未匹配时按顺序兜底,**任何阶段失败都不让单文档阻塞整批**。

### 1.3 多模态 Chunking 的上下文断裂

文本、图片、表格之间存在引用关系。固定窗口切 chunk 会割裂图文关系。

```
原文:
  段落A: "Q1营收表现亮眼..."          → chunk_1 (text)
  [营收趋势图.png]                    → chunk_2 (image, 2 路)
  段落B: "从上图可以看出..."          → chunk_3 (text)

问题: 用户问"Q1营收",chunk_3 被检索到,但说的是"上图",
     而图片 chunk_2 可能因向量距离远未被召回
```

**当前方案**:VL Caption 增强 + CLIP 双路让"以文搜图"更准。**未实现**:基于 MinerU 坐标的"召回时自动扩展相邻 chunk"。

### 1.4 模型/资源边界

| 模型 | 部署方式 | 显存 / 流量 |
|------|---------|------------|
| BGE Large zh v1.5 | Moark API(文本) | 0 本地 |
| Jina CLIP v2 | Moark API(图文) | 0 本地 |
| BCE reranker | Moark API | 0 本地 |
| Qwen2.5-VL-7B(可选) | Moark API | 0 本地 |
| Qwen3-0.6B(小模型) | Moark API | 0 本地 |
| MiniMax-M2.7(主 LLM) | 远端 API | 0 本地 |
| **本地合计** | — | **0** |

整个系统**零本地模型推理**,全部走云端 API。开发机只要 4GB 内存就能跑,Docker 部署也只要 8GB。

### 1.5 异步链路的可靠性

```
客户端上传 → API 写库 → Kafka → Worker 消费 → MinerU 异步 4 步 →
Text/Image Embedder → Milvus → 回写 MySQL
```

任何一个环节崩了都会让任务卡住。`parse_tasks.status` 状态机(`pending → processing → completed/failed/exhausted`) + 指数退避重试(30s → 60s → 90s) + `max_retries=3` 是核心保险。

### 1.6 小模型输出的稳定性

Qwen3-0.6B 输出 JSON 数组/多行文本,即便关 thinking 仍有概率出 think 标签残留或格式飘逸:

- **路由**:`_parse_kb_ids` 截取首 `[` 到末 `]`,再 `json.loads`,失败 → 全量 KB
- **改写**:行拆分 + `if question not in queries: insert(0, question)` 保证原问题一定在
- **命名**:clean `<think>` / 引号 / 书名号,失败 → 问题前 15 字

**核心原则:小模型失败不影响主流程**。

---

## 二、评测方案

### 2.1 评测层次总览

```
第四层: 工程性能(吞吐 / 延迟 / QPS)
          ↑
第三层: 消融实验(量化每组件贡献)
          ↑
第二层: RAG 端到端(RAGAS 自动化)
          ↑
第一层: 组件评测(Parser / Embedding / Rerank 离线评估)
```

### 2.2 第一层:组件评测

#### Parser 质量

**测试集**:20 份标注 PDF,覆盖 5 类 — 纯文本、混合图文、扫描件、学术论文、财务报告,每类 ≥ 3 份。

| 指标 | 测量方法 | 目标 |
|------|---------|------|
| 文本提取完整率 | MinerU 输出字符数 / 标注字符数 | > 95% |
| 表格结构准确率 | 行数 / 列数 / 合并单元格一致率 | > 90% |
| 图片召回率 | 提取图片数 / 标注图片数 | > 95% |
| v2 / v1 格式兼容率 | 双格式解析成功率 | > 99% |

#### Embedding 质量

| 指标 | 测试集 | 测量方法 | 目标 |
|------|--------|---------|------|
| 文本 Recall@10 | 200 对 (query, 相关段落) | BGE → Milvus → top-10 是否含目标 | > 90% |
| 图片 Recall@5 | 50 对 (文本 query, 相关图片) | Jina CLIP → Milvus → top-5 是否含目标 | > 85% |
| VL 增强增益 | 同 50 对 | 加 Qwen2.5-VL caption 后 Recall@5 | 基线 + 8% |

#### Rerank 增益

| 指标 | 基线(召回 score) | 目标(BCE-reranker) |
|------|------------------|---------------------|
| Recall@5 | 实测 | 基线 + 15% |
| MRR | 实测 | 基线 + 20% |
| 跨模态召回 | 不分桶(图片 0%) | 分桶 + 统一排序(图片 ≥ 25%) |

### 2.3 第二层:RAG 端到端

用 **RAGAS** 框架自动化评估。

**测试集**:50 条 `(question, ground_truth_answer, source_doc)` 三元组,覆盖单 KB / 跨 KB / 闲聊 / 不可答 4 类。

| 指标 | 含义 | 目标 |
|------|------|------|
| Faithfulness | 答案是否完全来自检索上下文 | > 0.90 |
| Answer Relevancy | 答案是否紧扣问题 | > 0.85 |
| Context Recall | 检索上下文覆盖 ground truth 所需信息 | > 0.85 |
| Context Precision | 检索结果中相关项排在前面的比例 | > 0.80 |

### 2.4 第三层:消融实验

| 实验 | 文本召回 | 图片召回 | Rerank | Query 改写 | 预期 Recall@5 |
|------|---------|---------|--------|-----------|---------------|
| E1 (基线) | ✓ | | | | — |
| E2 | ✓ | ✓ | | | E1 + 10% |
| E3 | ✓ | ✓ | ✓ | | E2 + 18% |
| E4 (全链路) | ✓ | ✓ | ✓ | ✓ | E3 + 8% |
| E5 (E4 + VL caption) | ✓ | ✓ | ✓ | ✓ | E4 + 5% |

**注意:代码中不存在 BM25 / 描述反查 / 多模态 LLM 重排等路径,消融表只能上下调开关,不能新增路径。**

### 2.5 第四层:工程性能

| 指标 | 测量方法 | 目标 |
|------|---------|------|
| 文档吞吐 | 100 页 PDF 从上传到入库的总耗时 | < 5 min |
| 查询延迟 P50 | 100 次查询,端到端延迟中位数 | < 2s |
| 查询延迟 P95 | 100 次查询,端到端 P95 | < 5s |
| 并发 QPS | 10 并发查询,统计 QPS | > 5 QPS |
| Kafka 消费延迟 | produce 到 consumer 拉取的时间 | < 500ms |
| 小模型路由延迟 | Qwen3-0.6B 一次 `route` 调用 | < 500ms |
| 小模型改写延迟 | Qwen3-0.6B 一次 `rewrite` 调用 | < 500ms |

---

## 三、风险优先级

| 优先级 | 风险 | 应对 |
|--------|------|------|
| P0 | PDF 解析质量不可控 | Parser 后加自动校验,不达标不入库,标 `failed` 重试 |
| P0 | 双 Embedding 空间不一致 | BCE-reranker 必须保留,文本/图片分桶再统一排序(已实现) |
| P1 | 测试集缺失 | 启动后用 10-20 份真实文档构建 RAGAS 测试集 |
| P1 | 图文关系断裂 | MinerU 坐标信息待接入,召回时自动扩展相邻 chunk |
| P2 | ImageEmbedder 不可用 | `validate()` 失败时跳过,主流程不阻塞(已实现) |
| P2 | VLCaptioner 不可用 | 降级到 MinerU caption(已实现) |
| P3 | 小模型格式飘逸 | 解析层容错 + 业务降级(已实现) |
