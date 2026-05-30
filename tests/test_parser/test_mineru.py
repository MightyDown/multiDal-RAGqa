from unittest.mock import MagicMock, mock_open, patch

from src.multidal.parser.mineru_parser import MinerUParser
from src.multidal.pipeline.base import PipelineContext
from src.multidal.schema.document import ParsedDocument


class TestMinerUParser:
    def test_validate_success(self):
        with patch("src.multidal.parser.mineru_parser.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            parser = MinerUParser()
            assert parser.validate() is True

    def test_validate_failure(self):
        with patch("src.multidal.parser.mineru_parser.requests.get") as mock_get:
            mock_get.side_effect = ConnectionError
            parser = MinerUParser()
            assert parser.validate() is False

    def test_process(self):
        mock_response = {
            "page_count": 3,
            "blocks": [
                {"text": "Q1营收增长15%", "page": 1, "type": "paragraph"},
                {"text": "如图表所示", "page": 2, "type": "paragraph"},
            ],
            "images": [
                {"page": 2, "caption": "营收趋势图", "label": "chart", "width": 800, "height": 600},
            ],
            "tables": [
                {"html": "<table><tr><td>100</td></tr></table>", "page": 3, "caption": "季度数据"},
            ],
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=b"fake pdf")),
            patch("src.multidal.parser.mineru_parser.requests.post") as mock_post,
        ):
            mock_post.return_value.json.return_value = mock_response
            mock_post.return_value.raise_for_status = MagicMock()

            parser = MinerUParser()
            ctx = PipelineContext(task_id="t1", file_path="/tmp/test.pdf", filename="test.pdf")
            result = parser.process(ctx)

            assert result.parsed is not None
            doc = result.parsed
            assert isinstance(doc, ParsedDocument)
            assert doc.page_count == 3
            assert len(doc.text_chunks) == 2
            assert len(doc.images) == 1
            assert len(doc.tables) == 1
            assert doc.images[0].caption == "营收趋势图"
            assert doc.tables[0].page == 3

    def test_process_file_not_found(self):
        parser = MinerUParser()
        ctx = PipelineContext(task_id="t1", file_path="/nonexistent.pdf", filename="nope.pdf")
        try:
            parser.process(ctx)
            assert False, "should have raised"
        except FileNotFoundError:
            pass
