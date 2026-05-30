import pytest
from pydantic import ValidationError

from src.multidal.schema.kb import KBCreateRequest, KBListResponse, KBResponse


class TestKBCreateRequest:
    def test_minimal(self):
        req = KBCreateRequest(name="财务报告")
        assert req.name == "财务报告"
        assert req.description == ""

    def test_name_too_short(self):
        with pytest.raises(ValidationError):
            KBCreateRequest(name="")

    def test_name_too_long(self):
        with pytest.raises(ValidationError):
            KBCreateRequest(name="x" * 129)

    def test_description_max_length(self):
        req = KBCreateRequest(name="kb", description="x" * 500)
        assert len(req.description) == 500

        with pytest.raises(ValidationError):
            KBCreateRequest(name="kb", description="x" * 513)


class TestKBResponse:
    def test_defaults(self):
        resp = KBResponse(kb_id="kb1", name="test")
        assert resp.description == ""
        assert resp.doc_count == 0

    def test_full(self):
        resp = KBResponse(kb_id="kb1", name="test", description="desc", doc_count=5)
        assert resp.doc_count == 5


class TestKBListResponse:
    def test_empty(self):
        resp = KBListResponse()
        assert resp.kbs == []
        assert resp.total == 0

    def test_with_items(self):
        resp = KBListResponse(kbs=[KBResponse(kb_id="k1", name="n1")], total=1)
        assert len(resp.kbs) == 1
