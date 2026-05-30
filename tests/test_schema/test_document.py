import pytest
from pydantic import ValidationError

from src.multidal.schema.document import ImageRegion, ParsedDocument, TableChunk, TextChunk


class TestImageRegion:
    def test_minimal(self):
        img = ImageRegion(image_id="img1", page=1)
        assert img.image_id == "img1"
        assert img.caption == ""
        assert img.label == ""
        assert img.width == 0
        assert img.height == 0
        assert img.image_path == ""

    def test_page_must_be_positive(self):
        with pytest.raises(ValidationError):
            ImageRegion(image_id="img1", page=0)

    def test_full_fields(self):
        img = ImageRegion(
            image_id="abc",
            page=3,
            bbox=(10.0, 20.0, 100.0, 200.0),
            caption="a chart",
            label="chart",
            width=800,
            height=600,
            image_path="/tmp/img.png",
        )
        assert img.bbox == (10.0, 20.0, 100.0, 200.0)
        assert img.width == 800
        assert img.image_path == "/tmp/img.png"


class TestTextChunk:
    def test_minimal(self):
        tc = TextChunk(chunk_id="c1", content="hello", page=1)
        assert tc.chunk_type == "paragraph"
        assert tc.bbox is None

    def test_content_min_length(self):
        with pytest.raises(ValidationError):
            TextChunk(chunk_id="c1", content="", page=1)

    def test_page_ge_1(self):
        with pytest.raises(ValidationError):
            TextChunk(chunk_id="c1", content="x", page=-1)


class TestTableChunk:
    def test_minimal(self):
        t = TableChunk(table_id="t1", html="<table></table>", page=1)
        assert t.caption == ""
        assert t.bbox is None

    def test_html_min_length(self):
        with pytest.raises(ValidationError):
            TableChunk(table_id="t1", html="", page=1)


class TestParsedDocument:
    def test_minimal(self):
        doc = ParsedDocument(doc_id="d1", filename="test.pdf", page_count=1)
        assert doc.text_chunks == []
        assert doc.images == []
        assert doc.tables == []
        assert doc.full_text == ""

    def test_with_chunks(self):
        doc = ParsedDocument(
            doc_id="d1",
            filename="f.pdf",
            page_count=5,
            text_chunks=[TextChunk(chunk_id="c1", content="hello", page=1)],
            images=[ImageRegion(image_id="i1", page=2)],
            tables=[TableChunk(table_id="t1", html="<table></table>", page=1)],
        )
        assert len(doc.text_chunks) == 1
        assert len(doc.tables) == 1

    def test_page_count_ge_1(self):
        with pytest.raises(ValidationError):
            ParsedDocument(doc_id="d1", filename="f.pdf", page_count=0)
