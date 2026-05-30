import pytest
from unittest.mock import MagicMock, patch

from src.multidal.embedder.text_embedder import TextEmbedder
from src.multidal.pipeline.base import PipelineContext


class TestTextEmbedderValidate:
    def test_validate_returns_bool(self):
        emb = TextEmbedder()
        result = emb.validate()
        assert isinstance(result, bool)


class TestTextEmbedderProcess:
    def test_requires_parsed(self):
        emb = TextEmbedder()
        ctx = PipelineContext(task_id="t1")
        with pytest.raises(ValueError, match="parsed is None"):
            emb.process(ctx)

    @patch("src.multidal.embedder.text_embedder.TextEmbedder._embed_batch")
    def test_empty_texts(self, mock_embed):
        from src.multidal.schema.document import ParsedDocument
        from src.multidal.schema.embedding import Embedding
        mock_embed.return_value = []
        emb = TextEmbedder()
        ctx = PipelineContext(task_id="t1", kb_id="kb1")
        ctx.parsed = ParsedDocument(doc_id="d1", filename="f.pdf", page_count=1, text_chunks=[], images=[])
        result = emb.process(ctx)
        assert result.embedded == []
