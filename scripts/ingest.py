"""CLI: 导入 PDF 到 multiDal 知识库。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from multidal.config import settings
from multidal.db.models import init_db
from multidal.db.repository import create_task
from multidal.queue.producer import KafkaProducer


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into multiDal")
    parser.add_argument("--file", required=True, help="Path to PDF file")
    parser.add_argument("--kb", default="default", help="Knowledge base ID")
    args = parser.parse_args()

    init_db()

    file_path = Path(args.file).resolve()
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)

    task = create_task(
        filename=file_path.name,
        file_path=str(file_path),
        kb_id=args.kb,
        file_size=file_path.stat().st_size,
    )

    producer = KafkaProducer()
    producer.send_parse_request(
        task_id=task.task_id,
        file_path=str(file_path),
        filename=file_path.name,
        kb_id=args.kb,
    )

    print(f"Task created: {task.task_id}")
    print(f"Status: {task.status}")
    print(f"Check: GET /ingest/{task.task_id}")


if __name__ == "__main__":
    main()
