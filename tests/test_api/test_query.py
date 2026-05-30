import pytest
from fastapi.testclient import TestClient

from src.multidal.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestQueryEndpoint:
    def test_query_empty_question_rejected(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_minimal(self, client):
        """最小查询请求，可能因无数据返回空或错误，但不应该 500。"""
        resp = client.post("/query", json={"question": "测试问题", "auto_route": True})
        assert resp.status_code != 500
