import pytest

from src.multidal.config.settings import Settings, settings


class TestSettings:
    def test_module_level_singleton(self):
        assert isinstance(settings, Settings)

    def test_defaults(self):
        s = Settings()
        assert s.top_k_recall == 10
        assert s.top_k_final == 5
        assert s.max_retries == 3
        assert "9092" in s.kafka_bootstrap_servers or s.kafka_bootstrap_servers == "localhost:9092"
        assert s.kafka_topic == "parse.request"
        assert isinstance(s.milvus_host, str) and len(s.milvus_host) > 0
        assert s.milvus_port == 19530
        assert s.log_level == "INFO"

    def test_pipeline_constraints(self):
        s = Settings()
        assert 1 <= s.top_k_recall <= 50
        assert 1 <= s.top_k_final <= 20
        assert 0 <= s.max_retries <= 10

    def test_milvus_port_range(self):
        s = Settings()
        assert 1 <= s.milvus_port <= 65535

    def test_project_root_is_path(self):
        from pathlib import Path
        assert isinstance(settings.project_root, Path)
        assert (settings.project_root / "src").exists()

    def test_extra_ignored(self):
        s = Settings(extra_field=999)  # type: ignore[call-arg]
        assert not hasattr(s, "extra_field")

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("TOP_K_RECALL", "25")
        s = Settings()
        assert s.top_k_recall == 25

    def test_db_path_default(self):
        s = Settings()
        assert s.db_path == "data/multidal.db"

    def test_text_embedding_defaults(self):
        s = Settings()
        assert s.text_embedding_dim == 1024

    def test_image_embedding_defaults(self):
        s = Settings()
        assert s.image_embedding_device == "cuda"
