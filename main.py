"""multiDal FastAPI 服务入口。"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录和 src 在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import uvicorn

from src.multidal.api.app import app
from src.multidal.config import settings


def main() -> None:
    uvicorn.run(
        "src.multidal.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
