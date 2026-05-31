from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_yaml() -> dict:
    path = _PROJECT_ROOT / "configs" / "settings.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


_cfg = _load_yaml()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    # ── Pipeline ──────────────────────────────────────────
    top_k_recall: int = Field(default=_cfg.get("pipeline", {}).get("top_k_recall", 10), ge=1, le=50)
    top_k_final: int = Field(default=_cfg.get("pipeline", {}).get("top_k_final", 5), ge=1, le=20)
    max_retries: int = Field(default=_cfg.get("pipeline", {}).get("max_retries", 3), ge=0, le=10)

    # ── Kafka ─────────────────────────────────────────────
    kafka_bootstrap_servers: str = Field(default=_cfg.get("kafka", {}).get("bootstrap_servers", "localhost:9092"))
    kafka_topic: str = Field(default=_cfg.get("kafka", {}).get("topic", "parse.request"))

    # ── Milvus ────────────────────────────────────────────
    milvus_host: str = Field(default=_cfg.get("milvus", {}).get("host", "localhost"))
    milvus_port: int = Field(default=_cfg.get("milvus", {}).get("port", 19530), ge=1, le=65535)

    # ── MinerU ────────────────────────────────────────────
    mineru_api_base: str = Field(default=_cfg.get("mineru", {}).get("api_base", ""))
    mineru_api_token: str = Field(default=_cfg.get("mineru", {}).get("api_token", ""))
    mineru_model_version: str = Field(default=_cfg.get("mineru", {}).get("model_version", "pipeline"))

    # ── DB ────────────────────────────────────────────────
    db_path: str = Field(default=_cfg.get("db", {}).get("path", "data/multidal.db"))
    mysql_host: str = Field(default=_cfg.get("db", {}).get("mysql_host", "localhost"))
    mysql_port: int = Field(default=_cfg.get("db", {}).get("mysql_port", 3306), ge=1, le=65535)
    mysql_user: str = Field(default=_cfg.get("db", {}).get("mysql_user", "root"))
    mysql_password: str = Field(default=_cfg.get("db", {}).get("mysql_password", "mysql"))
    mysql_database: str = Field(default=_cfg.get("db", {}).get("mysql_database", "multidal"))

    # ── Text Embedding ────────────────────────────────────
    text_embedding_api_base: str = Field(default=_cfg.get("text_embedding", {}).get("api_base", ""))
    text_embedding_model: str = Field(default=_cfg.get("text_embedding", {}).get("model", ""))
    text_embedding_dim: int = Field(default=_cfg.get("text_embedding", {}).get("dim", 1024))
    text_embedding_api_key: str = Field(default=_cfg.get("text_embedding", {}).get("api_key", ""))

    # ── Image Embedding ───────────────────────────────────
    image_embedding_name: str = Field(default=_cfg.get("image_embedding", {}).get("name", ""))
    image_embedding_api_base: str = Field(default=_cfg.get("image_embedding", {}).get("api_base", ""))
    image_embedding_api_key: str = Field(default=_cfg.get("image_embedding", {}).get("api_key", ""))
    image_embedding_model: str = Field(default=_cfg.get("image_embedding", {}).get("model", ""))
    image_embedding_device: str = Field(default=_cfg.get("image_embedding", {}).get("device", "cuda"))
    image_embedding_dim: int = Field(default=_cfg.get("image_embedding", {}).get("dim", 1024))

    # ── Reranker ──────────────────────────────────────────
    reranker_name: str = Field(default=_cfg.get("reranker", {}).get("name", ""))
    reranker_api_base: str = Field(default=_cfg.get("reranker", {}).get("api_base", ""))
    reranker_model: str = Field(default=_cfg.get("reranker", {}).get("model", ""))
    reranker_api_key: str = Field(default=_cfg.get("reranker", {}).get("api_key", ""))

    # ── VL Caption ────────────────────────────────────────
    vl_caption_name: str = Field(default=_cfg.get("vl_caption", {}).get("name", ""))
    vl_caption_api_base: str = Field(default=_cfg.get("vl_caption", {}).get("api_base", ""))
    vl_caption_api_key: str = Field(default=_cfg.get("vl_caption", {}).get("api_key", ""))
    vl_caption_model: str = Field(default=_cfg.get("vl_caption", {}).get("model", ""))

    # ── LLM ───────────────────────────────────────────────
    llm_api_key: str = Field(default=_cfg.get("llm", {}).get("api_key", ""))
    llm_base_url: str = Field(default=_cfg.get("llm", {}).get("base_url", ""))
    llm_model: str = Field(default=_cfg.get("llm", {}).get("model", ""))

    # ── Logging ───────────────────────────────────────────
    log_level: str = Field(default=_cfg.get("log", {}).get("level", "INFO"))

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT


settings = Settings()
