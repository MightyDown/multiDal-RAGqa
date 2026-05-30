from unittest.mock import MagicMock, patch

from src.multidal.embedder.registry import ModelRegistry
from src.multidal.embedder.text_embedder import TextEmbedder
from src.multidal.pipeline.base import PipelineContext
from src.multidal.schema.document import ParsedDocument, TextChunk


class TestModelRegistry:
    def test_singleton(self):
        a = ModelRegistry()
        b = ModelRegistry()
        assert a is b

    def test_put_get(self):
        reg = ModelRegistry()
        reg.put("test_model", MagicMock())
        assert reg.get("test_model") is not None
        assert "test_model" in reg.loaded

    def test_remove(self):
        reg = ModelRegistry()
        reg.put("m", MagicMock())
        reg.remove("m")
        assert reg.get("m") is None
        assert "m" not in reg.loaded


class TestTextEmbedder:
    def test_validate_success(self):
        with patch("src.multidal.embedder.text_embedder.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            embedder = TextEmbedder()
            assert embedder.validate() is True

    def test_process(self):
        mock_embedding = {
            "data": [{"embedding": [0.1] * 1024}],
        }
        doc = ParsedDocument(
            doc_id="d1",
            filename="test.pdf",
            page_count=1,
            text_chunks=[TextChunk(chunk_id="c1", content="hello", page=1)],
        )

        with patch("src.multidal.embedder.text_embedder.requests.post") as mock_post:
            mock_post.return_value.json.return_value = mock_embedding
            mock_post.return_value.raise_for_status = MagicMock()

            embedder = TextEmbedder()
            ctx = PipelineContext(task_id="t1", kb_id="kb1")
            ctx.parsed = doc
            result = embedder.process(ctx)

            assert result.embedded is not None
            assert len(result.embedded) == 1
            assert result.embedded[0].modality == "text"
            assert len(result.embedded[0].embedding.vector) == 1024
