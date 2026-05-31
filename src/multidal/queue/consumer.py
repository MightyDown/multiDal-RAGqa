from __future__ import annotations

import json
import logging

from confluent_kafka import Consumer, KafkaError

from src.multidal.config import settings
from src.multidal.db.models import init_db
from src.multidal.db.repository import update_task
from src.multidal.pipeline.base import PipelineContext
from src.multidal.schema.queue import ParseRequest
from src.multidal.schema.task import TaskStatus

logger = logging.getLogger(__name__)


class KafkaConsumer:
    def __init__(self) -> None:
        import socket
        self._consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": "multidal-workers-v2",
                "group.instance.id": f"worker-{socket.gethostname()}",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
                "max.poll.interval.ms": 600000,
                "session.timeout.ms": 30000,
            }
        )
        self._topic = settings.kafka_topic
        self._running = False

    def start(self, process_fn) -> None:
        """阻塞轮询，收到消息后调用 process_fn(req)。"""
        self._running = True
        self._consumer.subscribe([self._topic])
        logger.info("Consumer started, topic=%s", self._topic)

        try:
            while self._running:
                msg = self._consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("Kafka error: %s", msg.error())
                    continue

                try:
                    payload = json.loads(msg.value().decode("utf-8"))
                    req = ParseRequest(**payload)
                    logger.info("Received: task_id=%s", req.task_id)
                    process_fn(req)
                    self._consumer.commit(msg)
                except Exception:
                    logger.exception("Failed to process message, offset=%s", msg.offset())
        finally:
            self._consumer.close()
            logger.info("Consumer stopped")

    def stop(self) -> None:
        self._running = False


# ═══════════════════════════════════════════════════════════════
# Worker entry point
# ═══════════════════════════════════════════════════════════════

def _build_orchestrator():
    from src.multidal.embedder.image_embedder import ImageEmbedder
    from src.multidal.embedder.text_embedder import TextEmbedder
    from src.multidal.pipeline.orchestrator import Orchestrator
    from src.multidal.parser.mineru_parser import MinerUParser
    from src.multidal.store.milvus_store import MilvusStore

    stages = [MinerUParser(), TextEmbedder(), MilvusStore()]

    # 可选：图片向量化（需 CUDA + Jina CLIP 模型）+ VL caption
    try:
        from src.multidal.embedder.vl_captioner import VLCaptioner
        vl_captioner = VLCaptioner()
        if vl_captioner.validate():
            img_embedder = ImageEmbedder(vl_captioner=vl_captioner)
            logger.info("VL captioner enabled")
        else:
            img_embedder = ImageEmbedder()
            logger.warning("VL captioner not available, falling back to MinerU captions")
    except Exception:
        img_embedder = ImageEmbedder()
        logger.warning("VL captioner init failed, using MinerU captions")

    try:
        if img_embedder.validate():
            stages.insert(2, img_embedder)
            logger.info("Image embedder enabled")
        else:
            logger.warning("Image embedder not available, skipping")
    except Exception:
        logger.warning("Image embedder init failed, skipping")

    return Orchestrator(stages)


def _process_task(req: ParseRequest, orchestrator) -> None:
    task_id = req.task_id

    update_task(task_id, status=TaskStatus.PROCESSING.value, stage="parser")

    ctx = PipelineContext(
        task_id=task_id,
        kb_id=req.kb_id,
        file_path=req.file_path,
        filename=req.filename,
    )

    try:
        update_task(task_id, stage="parser")
        ctx = orchestrator.run(ctx)

        if ctx.parsed and ctx.parsed.page_count:
            update_task(
                task_id,
                page_count=ctx.parsed.page_count,
                full_text=ctx.parsed.full_text or "",
            )

        update_task(task_id, status=TaskStatus.COMPLETED.value, stage=None)
        logger.info("Task %s completed, chunks=%d", task_id, len(ctx.embedded or []))

    except Exception as e:
        logger.exception("Task %s failed", task_id)
        from src.multidal.db.repository import get_task
        task = get_task(task_id)
        retry = (task.retry_count + 1) if task else 1
        update_task(
            task_id,
            status=TaskStatus.FAILED.value,
            error_message=str(e)[:512],
            retry_count=retry,
        )

        if retry < settings.max_retries:
            logger.info("Retrying task %s (%d/%d)", task_id, retry, settings.max_retries)
            from src.multidal.queue.producer import KafkaProducer
            producer = KafkaProducer()
            producer.send_parse_request(
                task_id=req.task_id,
                file_path=req.file_path,
                filename=req.filename,
                kb_id=req.kb_id,
            )
        else:
            update_task(task_id, status=TaskStatus.EXHAUSTED.value)
            logger.error("Task %s exhausted after %d retries", task_id, retry)


if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _logs_dir = _Path(__file__).resolve().parents[3] / "logs"
    _logs_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(_logs_dir / "worker.log", encoding="utf-8"),
        ],
    )

    init_db()
    logger.info("Worker starting...")
    orchestrator = _build_orchestrator()

    consumer = KafkaConsumer()
    consumer.start(lambda req: _process_task(req, orchestrator))
