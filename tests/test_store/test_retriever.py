import pytest
from unittest.mock import MagicMock, patch

from src.multidal.schema.retrieval import RecallResult
from src.multidal.store.retriever import MultiPathRetriever


def _make_recall(chunk_id: str, score: float, kb_id: str = "kb1") -> RecallResult:
    return RecallResult(
        chunk_id=chunk_id, content="content", modality="text",
        source="milvus", score=score, kb_id=kb_id, doc_id="d1", page=1,
    )


class TestMultiPathRetriever:
    @pytest.fixture
    def mock_store(self):
        store = MagicMock()
        store.search.return_value = []
        return store

    def test_empty_results(self, mock_store):
        retriever = MultiPathRetriever(mock_store)
        results = retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1"])
        assert results == []

    def test_text_search_called(self, mock_store):
        def side_effect(coll_name, *args, **kwargs):
            if coll_name.endswith("_text"):
                return [_make_recall("c1", 0.9)]
            return [_make_recall("c2", 0.7)]
        mock_store.search.side_effect = side_effect
        retriever = MultiPathRetriever(mock_store)
        results = retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1"])
        assert len(results) == 2  # one from text, one from image

    def test_image_search_called(self, mock_store):
        mock_store.search.return_value = [_make_recall("img1", 0.7)]
        retriever = MultiPathRetriever(mock_store)
        retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1"])
        # image search: f"{kb}_image"
        mock_store.search.assert_any_call("kb1_image", [0.2] * 1024, top_k=10)

    def test_deduplication(self, mock_store):
        dup = _make_recall("c1", 0.9)
        mock_store.search.return_value = [dup]
        retriever = MultiPathRetriever(mock_store)
        results = retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1"])
        assert len(results) == 1

    def test_search_failure_graceful(self, mock_store):
        mock_store.search.side_effect = [Exception("boom"), [_make_recall("c1", 0.8)]]
        retriever = MultiPathRetriever(mock_store)
        results = retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1"])
        assert len(results) == 1

    def test_multi_kb(self, mock_store):
        mock_store.search.return_value = []
        retriever = MultiPathRetriever(mock_store)
        retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1", "kb2"])
        assert mock_store.search.call_count == 4

    def test_source_set_to_multi_path(self, mock_store):
        mock_store.search.return_value = [_make_recall("c1", 0.8)]
        retriever = MultiPathRetriever(mock_store)
        results = retriever.recall([0.1] * 1024, [0.2] * 1024, ["kb1"])
        for r in results:
            assert r.source == "multi_path"
