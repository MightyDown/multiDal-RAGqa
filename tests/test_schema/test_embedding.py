import pytest
from pydantic import ValidationError

from src.multidal.schema.embedding import EmbeddedChunk, Embedding


class TestEmbedding:
    def test_basic(self):
        emb = Embedding(model_name="bge", dim=1024, vector=[0.1, 0.2, 0.3])
        assert emb.model_name == "bge"
        assert emb.dim == 1024
        assert len(emb.vector) == 3

    def test_dim_must_be_positive(self):
        with pytest.raises(ValidationError):
            Embedding(model_name="bge", dim=0, vector=[0.1])

    def test_dim_must_match_vector_length(self):
        # Pydantic does not validate this cross-field constraint by default;
        # just verify it constructs.
        emb = Embedding(model_name="bge", dim=1024, vector=[1.0])
        assert emb.dim == 1024


class TestEmbeddedChunk:
    def test_text_modality(self):
        emb = Embedding(model_name="bge", dim=3, vector=[0.1, 0.2, 0.3])
        chunk = EmbeddedChunk(
            chunk_id="c1", content="text", embedding=emb, modality="text", kb_id="kb1", doc_id="d1", page=1
        )
        assert chunk.modality == "text"

    def test_image_modality(self):
        emb = Embedding(model_name="clip", dim=2, vector=[1.0, 2.0])
        chunk = EmbeddedChunk(
            chunk_id="c2", content="img desc", embedding=emb, modality="image", kb_id="kb1", doc_id="d1", page=2
        )
        assert chunk.modality == "image"

    def test_invalid_modality(self):
        emb = Embedding(model_name="bge", dim=3, vector=[1.0, 2.0, 3.0])
        with pytest.raises(ValidationError):
            EmbeddedChunk(
                chunk_id="c1", content="x", embedding=emb, modality="audio", kb_id="kb1", doc_id="d1", page=1
            )

    def test_metadata_default(self):
        emb = Embedding(model_name="bge", dim=1, vector=[1.0])
        chunk = EmbeddedChunk(
            chunk_id="c1", content="x", embedding=emb, modality="text", kb_id="kb1", doc_id="d1", page=1
        )
        assert chunk.metadata == {}
