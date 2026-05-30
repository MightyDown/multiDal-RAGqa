import pytest
from fastapi.testclient import TestClient

from src.multidal.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestStatusEndpoint:
    def test_nonexistent_task_returns_404(self, client):
        resp = client.get("/ingest/nonexistent_12345678")
        assert resp.status_code == 404
