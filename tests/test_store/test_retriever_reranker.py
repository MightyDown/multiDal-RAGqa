from unittest.mock import MagicMock, patch

from src.multidal.schema.retrieval import RecallResult, RerankResult
from src.multidal.store.reranker import Reranker
from src.multidal.store.retriever import MultiPathRetriever


def _make_candidates(n: int = 3) -> list[RecallResult]:
    return [
        RecallResult(
            chunk_id=f"c{i}", content=f"doc{i}", modality="text",
            source="dense_text", score=0.9 - i * 0.1,
            kb_id="kb1", doc_id="d1", page=i + 1,
        )
        for i in range(n)
    ]


class TestMultiPathRetriever:
    def test_recall_dedup(self):
        store = MagicMock()
        store.search.return_value = [
            RecallResult(
                chunk_id="c1", content="text", modality="text",
                source="milvus", score=0.85, kb_id="kb1", doc_id="d1", page=1,
            ),
            RecallResult(
                chunk_id="c1", content="text", modality="text",
                source="milvus", score=0.80, kb_id="kb1", doc_id="d1", page=1,
            ),
        ]
        retriever = MultiPathRetriever(store)
        results = retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1"])
        assert len(results) == 1
        assert results[0].score == 0.85


class TestReranker:
    def test_rerank_sorts_by_score(self):
        mock_response = {
            "results": [
                {"index": 0, "relevance_score": 0.2},
                {"index": 1, "relevance_score": 0.9},
                {"index": 2, "relevance_score": 0.5},
            ]
        }
        with patch("src.multidal.store.reranker.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            reranker = Reranker()
            candidates = _make_candidates(3)
            ranked = reranker.rerank("test query", candidates, top_k=2)

            assert len(ranked) == 2
            assert ranked[0].score == 0.9
            assert ranked[0].rank == 1
            assert isinstance(ranked[0], RerankResult)

    def test_rerank_empty(self):
        reranker = Reranker()
        assert reranker.rerank("q", []) == []

    def test_rerank_fallback_on_error(self):
        with patch("src.multidal.store.reranker.requests.post") as mock_post:
            mock_post.side_effect = ConnectionError

            reranker = Reranker()
            candidates = _make_candidates(2)
            ranked = reranker.rerank("q", candidates, top_k=2)
            assert len(ranked) == 2
            # fallback uses original scores
            assert ranked[0].score == 0.9
