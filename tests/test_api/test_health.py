import pytest
from fastapi.testclient import TestClient

from src.multidal.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "milvus" in data["components"]

    def test_health_has_required_components(self, client):
        response = client.get("/health")
        data = response.json()
        for key in ("milvus", "kafka", "mineru", "embedding", "reranker", "llm"):
            assert key in data["components"], f"Missing component: {key}"
