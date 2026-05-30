from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.multidal.api.routes import doc, health, ingest, kb, query, sessions, status
from src.multidal.config import settings
from src.multidal.db.models import init_db

app = FastAPI(title="multiDal", version="0.1.0")

app.include_router(ingest.router, tags=["ingest"], prefix="/api")
app.include_router(status.router, tags=["status"], prefix="/api")
app.include_router(query.router, tags=["query"], prefix="/api")
app.include_router(kb.router, tags=["kb"], prefix="/api")
app.include_router(doc.router, tags=["doc"], prefix="/api")
app.include_router(sessions.router, tags=["sessions"], prefix="/api")
app.include_router(health.router, tags=["health"], prefix="/api")

_static_dir = Path(__file__).resolve().parent.parent.parent.parent / "frontend" / "dist"
_assets_dir = _static_dir / "assets"
if _static_dir.exists():
    app.mount("/assets", StaticFiles(directory=str(_assets_dir)), name="assets")

_docs_dir = settings.project_root / "docs"
if _docs_dir.exists():
    app.mount("/raw", StaticFiles(directory=str(_docs_dir)), name="raw")


@app.get("/")
async def index():
    if _static_dir.exists():
        return FileResponse(_static_dir / "index.html")
    return {"message": "multiDal API — /docs for OpenAPI"}


@app.on_event("startup")
def on_startup() -> None:
    import logging as _logging
    _logs_dir = settings.project_root / "logs"
    _logs_dir.mkdir(parents=True, exist_ok=True)
    _fh = _logging.FileHandler(_logs_dir / "api.log", encoding="utf-8")
    _fh.setFormatter(_logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    _logging.getLogger().addHandler(_fh)
    _logging.getLogger("src.multidal").setLevel(_logging.INFO)

    init_db()
