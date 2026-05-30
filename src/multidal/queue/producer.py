from __future__ import annotations

import json
import logging

from confluent_kafka import Producer

from src.multidal.config import settings
from src.multidal.schema.queue import ParseRequest

logger = logging.getLogger(__name__)


class KafkaProducer:
    def __init__(self) -> None:
        self._producer = Producer({
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "socket.timeout.ms": 5000,
            "message.timeout.ms": 5000,
            "socket.connection.setup.timeout.ms": 3000,
        })
        self._topic = settings.kafka_topic

    def send_parse_request(self, task_id: str, file_path: str, filename: str, kb_id: str) -> None:
        msg = ParseRequest(
            task_id=task_id,
            file_path=file_path,
            filename=filename,
            kb_id=kb_id,
        )
        payload = msg.model_dump_json()
        self._producer.produce(self._topic, key=task_id, value=payload)
        remaining = self._producer.flush(5.0)  # 5s timeout, avoid blocking
        if remaining > 0:
            logger.warning("Kafka flush incomplete: %d messages pending for task %s", remaining, task_id)
        else:
            logger.info("Published parse request: task_id=%s filename=%s", task_id, filename)
