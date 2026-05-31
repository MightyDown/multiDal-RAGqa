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

POLL_INTERVAL = 5
MAX_POLL_ATTEMPTS = 120


class MinerUParser(Stage):
    """通过 MinerU 云 API (mineru.net) 解析 PDF。"""

    name = "parser"

    def __init__(self) -> None:
        self._api_base = settings.mineru_api_base
        self._token = settings.mineru_api_token
        self._model_version = settings.mineru_model_version

    def validate(self) -> bool:
        try:
            r = requests.get(
                f"{self._api_base}/api/v4/extract/task",
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=10,
            )
            return r.status_code != 401
        except Exception:
            logger.warning("MinerU cloud API not reachable at %s", self._api_base)
            return False

    def process(self, ctx: PipelineContext) -> PipelineContext:
        file_path = Path(ctx.file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {ctx.file_path}")

        result = self._parse(file_path, ctx.task_id)
        result.doc_id = ctx.task_id
        ctx.parsed = result
        return ctx

    def _parse(self, pdf_path: Path, task_id: str = "") -> ParsedDocument:
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
        if data.get("code") != 0:
            raise RuntimeError(f"MinerU upload URL request failed: {data}")

        batch_id = data["data"]["batch_id"]
        upload_url = data["data"]["file_urls"][0]
        logger.info("MinerU: batch_id=%s", batch_id)

        # Step 2: PUT 上传文件到 OSS
        logger.info("MinerU: uploading %d bytes to OSS", file_size)
        with open(pdf_path, "rb") as fh:
            r = requests.put(upload_url, data=fh, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"MinerU OSS upload failed: HTTP {r.status_code}")
        logger.info("MinerU: upload complete")

        # Step 3: 轮询 batch 结果
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

                # 每 6 次轮询（30 秒）输出一次进度
                if (attempt + 1) % 6 == 1:
                    elapsed = (attempt + 1) * POLL_INTERVAL
                    logger.info(
                        "MinerU batch %s: state=%s, attempt=%d/%d, elapsed=%ds",
                        batch_id, state, attempt + 1, MAX_POLL_ATTEMPTS, elapsed,
                    )
                else:
                    logger.debug("MinerU batch %s: %s (attempt %d)", batch_id, state, attempt + 1)

            except RuntimeError:
                raise
            except Exception:
                logger.debug("Batch poll error (attempt %d)", attempt + 1, exc_info=True)

        raise TimeoutError(
            f"MinerU batch {batch_id}: not completed after {MAX_POLL_ATTEMPTS} attempts"
        )

    def _build_document(self, z: zipfile.ZipFile, pdf_path: Path, task_id: str = "") -> ParsedDocument:
        """从 ZIP 中提取 content_list_v2.json 或 content_list.json + full.md，组装 ParsedDocument。"""

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

        # 找到 JSON 内容文件
        json_content = None
        md_content = ""
        for name in z.namelist():
            if name.endswith("_content_list_v2.json"):
                json_content = _json.loads(z.read(name).decode("utf-8"))
            elif name.endswith("_content_list.json") and json_content is None:
                json_content = _json.loads(z.read(name).decode("utf-8"))
            elif name == "full.md" or name.endswith("/full.md"):
                md_content = z.read(name).decode("utf-8")

        # 修正 markdown 中的图片路径，让前端能加载
        if md_content and task_id:
            md_content = md_content.replace("(images/", f"(/raw/{task_id}/images/")

        # 解析 v2 格式: [[{type, content, bbox}, ...], ...]
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

        # 降级：从 markdown 中拆分
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

        # 列出已提取的图片文件，供匹配
        _extracted_images: list[str] = []
        if _img_dir and _img_dir.exists():
            _extracted_images = sorted(
                [f.name for f in _img_dir.iterdir() if f.is_file()],
                key=lambda n: (int(n.split(".")[0]) if n.split(".")[0].isdigit() else 9999, n),
            )

        # 提取图片（扩大 block type 覆盖范围，覆盖 figure_image/chart/illustration 等）
        IMAGE_TYPES = {"image", "figure", "figure_image", "chart", "illustration", "inline_image", "page_image"}
        images: list[ImageRegion] = []
        _used_img_files: set[str] = set()  # 记录已通过 block 匹配到的图片文件名
        _img_idx = 0  # 顺序匹配计数器，用于 block 没有明确路径时兜底
        for block in blocks:
            if block.get("type") in IMAGE_TYPES:
                img_rel = block.get("image_path") or block.get("img_path") or ""
                img_name = img_rel.split("/")[-1] if img_rel else ""

                disk_path = ""
                if img_name and _img_dir and (_img_dir / img_name).exists():
                    disk_path = str(_img_dir / img_name)
                    _used_img_files.add(img_name)
                elif _img_idx < len(_extracted_images):
                    disk_path = str(_img_dir / _extracted_images[_img_idx])
                    _used_img_files.add(_extracted_images[_img_idx])

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

        # 兜底：把所有已提取但未被任何 block 匹配到的图片文件也入索引
        if _img_dir and _img_dir.exists():
            for img_name in _extracted_images:
                if img_name not in _used_img_files:
                    images.append(
                        ImageRegion(
                            image_id=uuid.uuid4().hex[:8],
                            page=0,  # 未知页，通过 caption 或文件名可追溯
                            caption=f"image file: {img_name}",
                            label="unmatched",
                            width=0,
                            height=0,
                            image_path=str(_img_dir / img_name),
                        )
                    )
            if _extracted_images and len(_extracted_images) != len(_used_img_files):
                logger.warning(
                    "MinerU: %d/%d image files unmatched by any block type, added via fallback",
                    len(_extracted_images) - len(_used_img_files),
                    len(_extracted_images),
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

        page_count = len(set(b.get("page_idx", 0) for b in blocks)) or len(text_chunks)

        return ParsedDocument(
            doc_id="",
            filename=pdf_path.name,
            file_path=str(pdf_path),
            page_count=page_count,
            text_chunks=text_chunks,
            images=images,
            tables=tables,
            full_text=md_content,
        )


def _extract_text_v2(block: dict) -> str:
    """从 v2 格式的嵌套 content 中提取文本。"""
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
    """从 v1 格式的 block 中提取文本（兼容旧格式）。"""
    return (block.get("text") or "").strip()
