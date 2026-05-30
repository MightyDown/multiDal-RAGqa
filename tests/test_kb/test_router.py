import pytest

from src.multidal.kb.router import _format_kb_list, _parse_kb_ids


class TestFormatKBList:
    def test_empty(self):
        from src.multidal.schema.kb import KBListResponse
        result = _format_kb_list(KBListResponse())
        assert result == ""

    def test_single(self):
        from src.multidal.schema.kb import KBListResponse, KBResponse
        kbs = KBListResponse(kbs=[KBResponse(kb_id="kb1", name="财务", description="年报")], total=1)
        result = _format_kb_list(kbs)
        assert "kb1" in result
        assert "财务" in result
        assert "年报" in result


class TestParseKBIds:
    def test_pure_json_array(self):
        assert _parse_kb_ids('["kb1", "kb2"]') == ["kb1", "kb2"]

    def test_json_with_markdown_wrapper(self):
        assert _parse_kb_ids('```json\n["kb1"]\n```') == ["kb1"]

    def test_json_with_text_around(self):
        assert _parse_kb_ids('返回: ["kb_finance", "kb_tech"] 请查收') == ["kb_finance", "kb_tech"]

    def test_single_id(self):
        assert _parse_kb_ids('["kb_finance"]') == ["kb_finance"]
