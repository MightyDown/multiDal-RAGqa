import io
import pytest
from fastapi.testclient import TestClient

from src.multidal.api.app import app


@pytest.fixture
def client():
    return TestClient(app)


def _make_pdf_bytes() -> bytes:
    """生成一个最小的有效 PDF。"""
    pdf = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\n"
        b"startxref\n190\n%%EOF"
    )
    return pdf


class TestIngestUpload:
    def test_upload_pdf_returns_task_id(self, client):
        # 先创建 KB
        cr = client.post("/kb/create", json={"name": "ingest_test_kb"})
        kb_id = cr.json()["kb_id"]

        pdf_bytes = _make_pdf_bytes()
        resp = client.post(
            f"/ingest?kb_id={kb_id}",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "pending"

        # cleanup
        client.delete(f"/kb/{kb_id}")

    def test_upload_without_file_rejected(self, client):
        resp = client.post("/ingest")
        assert resp.status_code == 422
