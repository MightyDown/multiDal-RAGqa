import io
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from PIL import Image

from src.multidal.embedder.image_embedder import ImageEmbedder, MAX_IMAGE_SIZE, JPEG_QUALITY
from src.multidal.pipeline.base import PipelineContext


class TestImageEmbedderStatic:
    def test_load_image_uri_empty_path(self):
        assert ImageEmbedder._load_image_uri("") is None

    def test_load_image_uri_nonexistent(self):
        assert ImageEmbedder._load_image_uri("/nonexistent/img.png") is None

    def test_load_image_uri_resize(self, tmp_path):
        img = Image.new("RGB", (500, 500), color="red")
        p = tmp_path / "large.png"
        img.save(p)
        result = ImageEmbedder._load_image_uri(str(p))
        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")

    def test_load_image_uri_small_no_resize(self, tmp_path):
        img = Image.new("RGB", (50, 50), color="blue")
        p = tmp_path / "small.png"
        img.save(p)
        result = ImageEmbedder._load_image_uri(str(p))
        assert result is not None
        assert result.startswith("data:image/jpeg;base64,")

    def test_make_chunk(self):
        from src.multidal.schema.document import ImageRegion, ParsedDocument
        img = ImageRegion(image_id="img1", page=3, image_path="/tmp/x.png", caption="chart")
        ctx = PipelineContext(task_id="t1", kb_id="kb1")
        ctx.parsed = ParsedDocument(doc_id="d1", filename="f.pdf", page_count=1)
        vec = [0.1, 0.2, 0.3]
        chunk = ImageEmbedder._make_chunk(img, "chart", vec, ctx)
        assert chunk.chunk_id == "img1"
        assert chunk.modality == "image"
        assert chunk.kb_id == "kb1"
        assert chunk.page == 3
        assert chunk.embedding.vector == vec

    def test_make_chunk_with_suffix(self):
        from src.multidal.schema.document import ImageRegion, ParsedDocument
        img = ImageRegion(image_id="img1", page=1, image_path="/tmp/x.png", caption="desc")
        ctx = PipelineContext(task_id="t1", kb_id="kb1")
        ctx.parsed = ParsedDocument(doc_id="d1", filename="f.pdf", page_count=1)
        chunk = ImageEmbedder._make_chunk(img, "desc", [0.1], ctx, suffix="_desc")
        assert chunk.chunk_id == "img1_desc"


class TestImageEmbedderValidate:
    def test_validate_returns_bool(self):
        emb = ImageEmbedder()
        result = emb.validate()
        assert isinstance(result, bool)


class TestImageEmbedderProcess:
    def test_no_images(self):
        emb = ImageEmbedder()
        ctx = PipelineContext(task_id="t1")
        from src.multidal.schema.document import ParsedDocument
        ctx.parsed = ParsedDocument(doc_id="d1", filename="f.pdf", page_count=1, text_chunks=[], images=[])
        result = emb.process(ctx)
        assert result.embedded == [] or result.embedded is None

    def test_requires_parsed(self):
        emb = ImageEmbedder()
        ctx = PipelineContext(task_id="t1")
        with pytest.raises(ValueError, match="parsed is None"):
            emb.process(ctx)
