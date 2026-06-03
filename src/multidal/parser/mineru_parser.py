"""MinerU 云 API 解析器。

本模块实现 ``MinerUParser``(继承自 ``pipeline.Stage``),通过 mineru.net 提供的
异步 PDF 解析服务完成 PDF -> 结构化文档的转换。

完整调用流程:
    1. 申请上传 URL(``/api/v4/file-urls/batch``)。
    2. PUT 上传 PDF 到返回的 OSS URL。
    3. 轮询 batch 状态(``/api/v4/extract-results/batch/{batch_id}``)。
    4. 下载 full_zip_url 提供的 ZIP 包,解压后组装 ``ParsedDocument``。

ZIP 包内关键文件:
    - ``*_content_list.json`` / ``*_content_list_v2.json``: 结构化块列表(嵌套列表)。
    - ``full.md``: 完整 Markdown 文本(用于 LLM 兜底)。
    - ``images/*``: 抽出的图片文件(按出现顺序编号)。
"""

from __future__ import annotations

import io
import json as _json
import logging
import time
import uuid
import zipfile
from pathlib import Path

import requests

from src.multidal.config import settings
from src.multidal.pipeline.base import PipelineContext, Stage
from src.multidal.schema.document import ImageRegion, ParsedDocument, TableChunk, TextChunk

logger = logging.getLogger(__name__)

# 轮询间隔(秒):MinerU 单文档解析通常 30-90s,5s 间隔能在及时性与 API 限流间平衡。
POLL_INTERVAL = 5
# 最大轮询次数:5s × 120 = 600s = 10 分钟,超过则认为本次解析超时失败。
MAX_POLL_ATTEMPTS = 120


class MinerUParser(Stage):
    """通过 MinerU 云 API (mineru.net) 解析 PDF 的流水线阶段。

    Attributes:
        name: 阶段名,固定为 ``"parser"``,供 Orchestrator 日志使用。
        _api_base: MinerU API 根地址(从配置读取)。
        _token: Bearer Token(从配置读取)。
        _model_version: 解析模型版本(从配置读取,例如 ``"vlm"``)。
    """

    name = "parser"

    def __init__(self) -> None:
        """从全局配置读取 MinerU API 凭据与模型版本。"""
        self._api_base = settings.mineru_api_base
        self._token = settings.mineru_api_token
        self._model_version = settings.mineru_model_version

    def validate(self) -> bool:
        """通过探测 API 验证 Token 是否有效。

        策略:访问 ``/api/v4/extract/task``,只要不返回 ``401`` 即认为可用。
        捕获所有网络异常(超时、连接失败等)并降级为 False。

        Returns:
            bool: True 表示凭据有效且 API 可达;False 表示不可用。
        """
        try:
            r = requests.get(
                f"{self._api_base}/api/v4/extract/task",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            # 仅 401 视为鉴权失败;其他状态码(如 404/500)说明接口可达但参数问题
            return r.status_code != 401
        except Exception:
            # 网络层异常(超时/拒连/DNS 失败)统一降级
            logger.warning("MinerU cloud API not reachable at %s", self._api_base)
            return False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        """执行解析并将结果写入 ctx.parsed。

        步骤:
            1. 校验 PDF 是否存在。
            2. 调用 ``_parse`` 完成上传 + 轮询 + 下载。
            3. 将 ``doc_id`` 绑定为 ``ctx.task_id``(与 MySQL 任务一一对应)。
            4. 写回 ``ctx.parsed``。

        Args:
            ctx: 流水线上下文,需含 ``file_path`` 与 ``task_id``。

        Returns:
            PipelineContext: 已填充 ``parsed`` 字段的上下文。

        Raises:
            FileNotFoundError: PDF 文件不存在。
            RuntimeError: API 返回非 0 code 或 OSS 上传失败等。
        """
        file_path = Path(ctx.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {ctx.file_path}")

        result = self._parse(file_path, ctx.task_id)
        result.doc_id = ctx.task_id
        ctx.parsed = result
        return ctx

    def _parse(self, pdf_path: Path, task_id: str = "") -> ParsedDocument:
        """完整解析单个 PDF:申请 URL -> 上传 -> 轮询 -> 下载 -> 组装。

        Args:
            pdf_path: 已落盘的 PDF 文件路径。
            task_id: 任务 ID,用于定位图片落地目录(``{pdf_dir}/{task_id}/images/``)。

        Returns:
            ParsedDocument: 组装好的解析结果(含 text_chunks、images、tables、full_text)。
        """
        file_size = pdf_path.stat().st_size

        # Step 1: 申请上传 URL
        logger.info("MinerU: requesting upload URL for %s (%d bytes)", pdf_path.name, file_size)
        r = requests.post(
            f"{self._api_base}/api/v4/file-urls/batch",
            json={
                "files": [{"name": pdf_path.name, "size": file_size}],
                "model_version": self._model_version,
            },
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        # MinerU API 约定 code=0 表示业务成功,其他值均为业务错误
        if data.get("code") != 0:
            raise RuntimeError(f"MinerU upload URL request failed: {data}")

        batch_id = data["data"]["batch_id"]
        upload_url = data["data"]["file_urls"][0]
        logger.info("MinerU: batch_id=%s", batch_id)

        # Step 2: PUT 上传文件到 OSS(MinerU 签发的临时 URL,5 分钟内有效)
        logger.info("MinerU: uploading %d bytes to OSS", file_size)
        with open(pdf_path, "rb") as fh:
            r = requests.put(upload_url, data=fh, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"MinerU OSS upload failed: HTTP {r.status_code}")
        logger.info("MinerU: upload complete")

        # Step 3: 轮询 batch 结果,直到 done/failed 或超时
        result = self._poll_batch(batch_id)

        # Step 4: 下载并解压结果 ZIP
        full_zip_url = result.get("full_zip_url", "")
        if not full_zip_url:
            raise RuntimeError(f"MinerU batch {batch_id}: no full_zip_url in result")

        logger.info("MinerU: downloading result ZIP")
        r = requests.get(full_zip_url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))

        return self._build_document(z, pdf_path, task_id)

    def _poll_batch(self, batch_id: str) -> dict:
        """轮询 MinerU batch 状态,直到完成或失败。

        轮询策略:
            - 每 ``POLL_INTERVAL`` 秒请求一次,最多 ``MAX_POLL_ATTEMPTS`` 次。
            - 状态机:``done`` -> 返回结果;``failed`` / ``error`` -> 抛错;
              其他(``pending`` / ``running`` / ...) -> 继续轮询。
            - 每 6 次(30 秒)在 INFO 级别打印一次进度,其余 DEBUG,避免日志刷屏。

        Args:
            batch_id: MinerU 返回的批次 ID。

        Returns:
            dict: MinerU 返回的结果对象(含 ``full_zip_url`` 等字段)。

        Raises:
            RuntimeError: batch 进入 ``failed`` / ``error`` 状态。
            TimeoutError: 超过最大轮询次数仍未完成。
        """
        logger.info("MinerU batch %s: waiting for result...", batch_id)
        for attempt in range(MAX_POLL_ATTEMPTS):
            time.sleep(POLL_INTERVAL)
            try:
                r = requests.get(
                    f"{self._api_base}/api/v4/extract-results/batch/{batch_id}",
                    headers={"Authorization": f"Bearer {self._token}"},
                    timeout=15,
                )
                if r.status_code != 200:
                    # 偶发 HTTP 错误(如 502)继续轮询,不当成失败
                    logger.debug("Batch poll HTTP %d (attempt %d)", r.status_code, attempt + 1)
                    continue

                data = r.json()
                if data.get("code") != 0:
                    logger.debug("Batch poll code %s (attempt %d)", data.get("code"), attempt + 1)
                    continue

                results = data.get("data", {}).get("extract_result", [])
                if not results:
                    continue

                result = results[0]
                state = result.get("state", "")

                if state == "done":
                    logger.info("MinerU batch %s: done (attempt %d)", batch_id, attempt + 1)
                    return result

                if state in ("failed", "error"):
                    raise RuntimeError(
                        f"MinerU batch {batch_id} failed: {result.get('err_msg', '')}"
                    )

                # 每 6 次轮询(30 秒)输出一次进度,便于运维观察
                if (attempt + 1) % 6 == 1:
                    elapsed = (attempt + 1) * POLL_INTERVAL
                    logger.info(
                        "MinerU batch %s: state=%s, attempt=%d/%d, elapsed=%ds",
                        batch_id, state, attempt + 1, MAX_POLL_ATTEMPTS, elapsed,
                    )
                else:
                    logger.debug("MinerU batch %s: %s (attempt %d)", batch_id, state, attempt + 1)

            except RuntimeError:
                # 业务级错误向上抛,不吞掉
                raise
            except Exception:
                # 网络异常仅 DEBUG,不打断轮询
                logger.debug("Batch poll error (attempt %d)", attempt + 1, exc_info=True)

        raise TimeoutError(
            f"MinerU batch {batch_id}: not completed after {MAX_POLL_ATTEMPTS} attempts"
        )

    def _build_document(self, z: zipfile.ZipFile, pdf_path: Path, task_id: str = "") -> ParsedDocument:
        """从 MinerU 结果 ZIP 中提取内容并组装 ``ParsedDocument``。

        处理流程:
            1. 抽取 ``images/*`` 到 ``{pdf_dir}/{task_id}/images/``(供前端回显)。
            2. 读取 ``*_content_list_v2.json``(优先)或 ``*_content_list.json``(兼容)。
            3. 读取 ``full.md`` 作为完整 Markdown 文本,并改写其中的图片路径。
            4. 解析所有 block:
                - 文本/标题块 -> ``TextChunk``;
                - 图片块 -> ``ImageRegion``(含磁盘路径匹配);
                - 表格块 -> ``TableChunk``。
            5. 兜底:若 JSON 解析失败,从 markdown 段落中切分文本块。
            6. 兜底:未被 block 匹配到的图片文件也加入索引(防止漏召)。

        Args:
            z: 内存中已打开的 ZIP 对象。
            pdf_path: 原 PDF 路径,用于推导图片落地目录。
            task_id: 任务 ID,空字符串表示不落地图片(测试场景)。

        Returns:
            ParsedDocument: 组装好的文档对象。
        """
        # 提取图片到 docs/{task_id}/images/
        _img_dir: Path | None = None
        if task_id:
            _img_dir = pdf_path.parent / f"{task_id}" / "images"
            _img_dir.mkdir(parents=True, exist_ok=True)
            for name in z.namelist():
                if name.startswith("images/") and not name.endswith("/"):
                    fname = name.split("/", 1)[1]
                    (_img_dir / fname).write_bytes(z.read(name))
            logger.info("MinerU: saved %d images to %s", len(list(_img_dir.glob("*"))), _img_dir)

        # 找到 JSON 内容文件(v2 优先,v1 兼容)
        json_content = None
        md_content = ""
        for name in z.namelist():
            if name.endswith("_content_list_v2.json"):
                json_content = _json.loads(z.read(name).decode("utf-8"))
            elif name.endswith("_content_list.json") and json_content is None:
                json_content = _json.loads(z.read(name).decode("utf-8"))
            elif name == "full.md" or name.endswith("/full.md"):
                md_content = z.read(name).decode("utf-8")

        # 修正 markdown 中的图片路径,让前端能加载
        # 原路径: (images/xxx.jpg) -> 新路径: (/raw/{task_id}/images/xxx.jpg)
        if md_content and task_id:
            md_content = md_content.replace("(images/", f"(/raw/{task_id}/images/")

        # 解析 v2 格式: [[{type, content, bbox}, ...], ...] (外层按页分组)
        blocks = []
        if isinstance(json_content, list):
            for page_blocks in json_content:
                if isinstance(page_blocks, list):
                    blocks.extend(page_blocks)

        # 构建文本块
        text_chunks: list[TextChunk] = []
        for block in blocks:
            text = _extract_text_v2(block) or _extract_text_v1(block)
            if not text:
                continue
            text_chunks.append(
                TextChunk(
                    chunk_id=uuid.uuid4().hex[:8],
                    content=text,
                    page=block.get("page_idx", 0) + 1,
                    chunk_type=block.get("type", "paragraph"),
                )
            )

        # 降级:从 markdown 中拆分(无 JSON 时)
        if not text_chunks and md_content:
            paragraphs = [p.strip() for p in md_content.split("\n\n") if p.strip()]
            page = 1
            for p in paragraphs:
                text_chunks.append(
                    TextChunk(
                        chunk_id=uuid.uuid4().hex[:8],
                        content=p,
                        page=page,
                        chunk_type="paragraph",
                    )
                )
                page += 1

        # 列出已提取的图片文件,供匹配
        _extracted_images: list[str] = []
        if _img_dir and _img_dir.exists():
            _extracted_images = sorted(
                [f.name for f in _img_dir.iterdir() if f.is_file()],
                # 数字开头的按数字升序(如 0.jpg, 1.jpg, 10.jpg),
                # 非数字的兜底排到最后,保证与 block 顺序大致一致
                key=lambda n: (int(n.split(".")[0]) if n.split(".")[0].isdigit() else 9999, n),
            )

        # 提取图片(扩大 block type 覆盖范围,覆盖 figure_image/chart/illustration 等)
        # MinerU 不同版本/文档类型可能产生不同的 type 字符串,这里宽松匹配
        IMAGE_TYPES = {"image", "figure", "figure_image", "chart", "illustration", "inline_image", "page_image"}
        images: list[ImageRegion] = []
        _img_idx = 0  # 顺序匹配计数器,用于 block 没有明确路径时兜底
        for block in blocks:
            if block.get("type") in IMAGE_TYPES:
                img_rel = block.get("image_path") or block.get("img_path") or ""
                img_name = img_rel.split("/")[-1] if img_rel else ""

                disk_path = ""
                if img_name and _img_dir and (_img_dir / img_name).exists():
                    disk_path = str(_img_dir / img_name)
                elif _img_idx < len(_extracted_images):
                    # block 未声明路径时,按出现顺序兜底匹配一个未占用的图片
                    disk_path = str(_img_dir / _extracted_images[_img_idx])

                _img_idx += 1

                images.append(
                    ImageRegion(
                        image_id=uuid.uuid4().hex[:8],
                        page=block.get("page_idx", 0) + 1,
                        caption=_extract_text_v2(block) or "",
                        label=block.get("type", ""),
                        width=0,
                        height=0,
                        image_path=disk_path,
                    )
                )

        # 兜底:把所有已提取但未被任何 block 匹配到的图片文件也入索引
        # 逻辑:images 列表里的 image_path 与 _extracted_images 比对
        _indexed_paths = {img.image_path.split("/")[-1] for img in images if img.image_path}
        if _img_dir and _img_dir.exists() and _extracted_images:
            unmatched = [f for f in _extracted_images if f not in _indexed_paths]
            if unmatched:
                logger.warning(
                    "MinerU: %d/%d image files unmatched by any block type, adding via fallback",
                    len(unmatched),
                    len(_extracted_images),
                )
                for img_name in unmatched:
                    images.append(
                        ImageRegion(
                            image_id=uuid.uuid4().hex[:8],
                            page=1,  # 无法确定页码,默认 1
                            caption=f"image file: {img_name}",
                            label="unmatched",
                            width=0,
                            height=0,
                            image_path=str(_img_dir / img_name),
                        )
                    )

        # 提取表格
        tables: list[TableChunk] = []
        for block in blocks:
            if block.get("type") == "table":
                html = _extract_text_v2(block) or ""
                if not html:
                    continue
                tables.append(
                    TableChunk(
                        table_id=uuid.uuid4().hex[:8],
                        html=html,
                        page=block.get("page_idx", 0) + 1,
                        caption="",
                    )
                )

        # 页数:优先用 block 中出现过的最大 page_idx + 1;无 block 则用文本块数(粗略)
        page_count = len(set(b.get("page_idx", 0) for b in blocks)) or len(text_chunks)

        return ParsedDocument(
            doc_id="",  # 留空,由 process() 写入 task_id
            filename=pdf_path.name,
            file_path=str(pdf_path),
            page_count=page_count,
            text_chunks=text_chunks,
            images=images,
            tables=tables,
            full_text=md_content,
        )


def _extract_text_v2(block: dict) -> str:
    """从 v2 格式的 block 中提取文本。

    v2 结构示意::

        block = {
            "type": "text",
            "page_idx": 0,
            "content": {
                "paragraph_content": [{"content": "句子1"}, {"content": "句子2"}]
            }
        }

    提取策略:遍历 ``content`` 字典,找到第一个 list/str 值并拼接。
    不同 block type 对应的 content 键名不同(paragraph_content / title_content /
    table_content 等),故统一按结构匹配。

    Args:
        block: 单个 block 字典。

    Returns:
        str: 提取出的文本,无文本时返回空串。
    """
    content = block.get("content", {})
    if not isinstance(content, dict):
        return ""
    # 尝试各种 content 类型: paragraph_content, title_content, table_content 等
    for key, value in content.items():
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    parts.append(item.get("content", ""))
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        elif isinstance(value, str):
            return value
    return ""


def _extract_text_v1(block: dict) -> str:
    """从 v1 格式的 block 中提取文本(兼容旧版 MinerU 输出)。

    v1 格式较简单,直接读 ``block["text"]`` 字段即可。

    Args:
        block: 单个 block 字典。

    Returns:
        str: 文本内容(已 strip),无文本时返回空串。
    """
    return (block.get("text") or "").strip()
