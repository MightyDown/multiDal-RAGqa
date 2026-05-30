import pytest

from src.multidal.kb.manager import KBManager


class TestKBManager:
    def test_create_returns_response(self):
        mgr = KBManager()
        resp = mgr.create("test_kb", "desc")
        assert resp.name == "test_kb"
        assert resp.description == "desc"
        assert resp.kb_id != ""

    def test_list_all_includes_created(self):
        mgr = KBManager()
        mgr.create("kb_a")
        result = mgr.list_all()
        assert result.total >= 1
        assert any(kb.name == "kb_a" for kb in result.kbs)

    def test_list_all_ids(self):
        mgr = KBManager()
        mgr.create("kb_b")
        ids = mgr.list_all_ids()
        assert len(ids) >= 1

    def test_delete_returns_true(self):
        mgr = KBManager()
        resp = mgr.create("to_delete")
        assert mgr.delete(resp.kb_id) is True

    def test_delete_nonexistent(self):
        mgr = KBManager()
        assert mgr.delete("nonexistent_kb") is False

    def test_create_default_description(self):
        mgr = KBManager()
        resp = mgr.create("minimal")
        assert resp.description == ""
