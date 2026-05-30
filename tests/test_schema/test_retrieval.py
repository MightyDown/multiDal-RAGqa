import pytest
from pydantic import ValidationError

from src.multidal.schema.retrieval import RecallResult, RerankResult, SearchRequest, SearchResult


class TestSearchRequest:
    def test_minimal(self):
        req = SearchRequest(query="test", kb_ids=["kb1"])
        assert req.modality == "hybrid"
        assert req.top_k == 10

    def test_query_min_length(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="", kb_ids=["kb1"])

    def test_kb_ids_min_length(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", kb_ids=[])

    def test_invalid_modality(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", kb_ids=["kb1"], modality="unknown")

    def test_top_k_range(self):
        with pytest.raises(ValidationError):
            SearchRequest(query="test", kb_ids=["kb1"], top_k=0)
        with pytest.raises(ValidationError):
            SearchRequest(query="test", kb_ids=["kb1"], top_k=51)


class TestSearchResult:
    def test_basic(self):
        sr = SearchResult(
            chunk_id="c1", content="text", modality="text", kb_id="kb1", doc_id="d1", page=1, score=0.95
        )
        assert sr.score == 0.95

    def test_score_range(self):
        with pytest.raises(ValidationError):
            SearchResult(
                chunk_id="c1", content="x", modality="text", kb_id="k1", doc_id="d1", page=1, score=1.5
            )
        with pytest.raises(ValidationError):
            SearchResult(
                chunk_id="c1", content="x", modality="text", kb_id="k1", doc_id="d1", page=1, score=-0.1
            )

    def test_modality_pattern(self):
        with pytest.raises(ValidationError):
            SearchResult(
                chunk_id="c1", content="x", modality="audio", kb_id="k1", doc_id="d1", page=1, score=0.5
            )


class TestRecallResult:
    def test_basic(self):
        rr = RecallResult(
            chunk_id="c1",
            content="text",
            modality="text",
            source="dense_text",
            score=0.8,
            kb_id="kb1",
            doc_id="d1",
            page=1,
        )
        assert rr.source == "dense_text"
        assert rr.image_path is None

    def test_with_image_path(self):
        rr = RecallResult(
            chunk_id="img1",
            content="a chart showing growth",
            modality="image",
            source="dense_image",
            score=0.85,
            kb_id="kb1",
            doc_id="d1",
            page=3,
            image_path="docs/task123/images/img_001.jpg",
        )
        assert rr.modality == "image"
        assert rr.image_path == "docs/task123/images/img_001.jpg"

    def test_image_path_null_for_text(self):
        rr = RecallResult(
            chunk_id="c1",
            content="some text content",
            modality="text",
            source="dense_text",
            score=0.7,
            kb_id="kb1",
            doc_id="d1",
            page=2,
        )
        assert rr.image_path is None


class TestRerankResult:
    def test_basic(self):
        rr = RerankResult(
            chunk_id="c1",
            content="text",
            modality="text",
            score=0.9,
            rank=1,
            kb_id="kb1",
            doc_id="d1",
            page=1,
        )
        assert rr.rank == 1

    def test_rank_ge_1(self):
        with pytest.raises(ValidationError):
            RerankResult(
                chunk_id="c1",
                content="x",
                modality="text",
                score=0.5,
                rank=0,
                kb_id="kb1",
                doc_id="d1",
                page=1,
            )
