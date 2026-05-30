import pytest
from fastapi.testclient import TestClient

from src.multidal.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


class TestKBCreate:
    def test_create_minimal(self, client):
        resp = client.post("/kb/create", json={"name": "api_test_kb"})
        assert resp.status_code == 200
        data = resp.json()
        assert "kb_id" in data
        assert data["name"] == "api_test_kb"
        # cleanup
        if "kb_id" in data:
            client.delete(f"/kb/{data['kb_id']}")

    def test_create_with_description(self, client):
        resp = client.post("/kb/create", json={"name": "api_desc_kb", "description": "测试库"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["description"] == "测试库"
        if "kb_id" in data:
            client.delete(f"/kb/{data['kb_id']}")

    def test_create_empty_name_rejected(self, client):
        resp = client.post("/kb/create", json={"name": ""})
        assert resp.status_code == 422


class TestKBList:
    def test_list_returns_array(self, client):
        # 先创建一个 KB
        cr = client.post("/kb/create", json={"name": "list_test_kb"})
        kb_id = cr.json().get("kb_id", "")

        resp = client.get("/kb/list")
        assert resp.status_code == 200
        data = resp.json()
        assert "kbs" in data
        assert "total" in data
        assert isinstance(data["kbs"], list)
        assert data["total"] >= 1

        if kb_id:
            client.delete(f"/kb/{kb_id}")


class TestKBDelete:
    def test_delete_existing(self, client):
        cr = client.post("/kb/create", json={"name": "del_test"})
        kb_id = cr.json()["kb_id"]
        resp = client.delete(f"/kb/{kb_id}")
        assert resp.status_code == 200

    def test_delete_nonexistent(self, client):
        resp = client.delete("/kb/nonexistent_12345")
        assert resp.status_code == 404
