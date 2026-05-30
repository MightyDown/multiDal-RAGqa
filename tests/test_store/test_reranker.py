import pytest

from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.reranker import Reranker


def _make_recall(chunk_id: str, content: str, score: float) -> RecallResult:
    return RecallResult(
        chunk_id=chunk_id, content=content, modality="text",
        source="milvus", score=score, kb_id="kb1", doc_id="d1", page=1,
    )


class TestReranker:
    def test_empty_candidates(self):
        rr = Reranker()
        result = rr.rerank("query", [])
        assert result == []

    def test_fallback_on_api_failure(self):
        rr = Reranker()
        candidates = [_make_recall("c1", "hello", 0.8), _make_recall("c2", "world", 0.6)]
        result = rr.rerank("query", candidates, top_k=2)
        assert len(result) == 2
        assert result[0].rank == 1
        assert result[1].rank == 2

    def test_top_k_truncation(self):
        rr = Reranker()
        candidates = [_make_recall(f"c{i}", f"text{i}", 1.0 - i * 0.1) for i in range(10)]
        result = rr.rerank("query", candidates, top_k=3)
        assert len(result) == 3
        ranks = [r.rank for r in result]
        assert ranks == [1, 2, 3]

    def test_scores_descending(self):
        rr = Reranker()
        candidates = [
            _make_recall("c1", "a", 0.3),
            _make_recall("c2", "b", 0.9),
            _make_recall("c3", "c", 0.5),
        ]
        result = rr.rerank("query", candidates)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_score_parsing_list(self):
        """_score() should handle plain float array from Moark API."""
        rr = Reranker()
        candidates = [_make_recall("c1", "text", 0.5), _make_recall("c2", "more", 0.3)]
        scores = rr._score("q", candidates)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, (int, float)) for s in scores)

    def test_validate(self):
        rr = Reranker()
        result = rr.validate()
        assert isinstance(result, bool)
