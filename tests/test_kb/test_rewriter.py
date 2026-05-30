import pytest


class TestQueryRewriter:
    def test_init_does_not_require_llm(self):
        """QueryRewriter 构造不需要 LLM 可达。"""
        from src.multidal.kb.rewriter import QueryRewriter
        rw = QueryRewriter()
        assert rw is not None
