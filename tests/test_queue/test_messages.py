from unittest.mock import MagicMock, patch

from src.multidal.queue.consumer import KafkaConsumer
from src.multidal.queue.producer import KafkaProducer
from src.multidal.schema.queue import ParseRequest


class TestParseRequest:
    def test_serialize_deserialize(self):
        req = ParseRequest(
            task_id="abc123",
            file_path="/data/test.pdf",
            filename="test.pdf",
            kb_id="kb_finance",
        )
        payload = req.model_dump_json()
        reloaded = ParseRequest.model_validate_json(payload)
        assert reloaded.task_id == "abc123"
        assert reloaded.filename == "test.pdf"
        assert reloaded.kb_id == "kb_finance"


class TestKafkaProducer:
    def test_send_parse_request(self):
        with patch("src.multidal.queue.producer.Producer") as MockProducer:
            mock = MagicMock()
            mock.flush.return_value = 0  # no pending messages
            MockProducer.return_value = mock

            producer = KafkaProducer()
            producer.send_parse_request("t1", "/tmp/a.pdf", "a.pdf", "kb1")

            mock.produce.assert_called_once()
            call_args = mock.produce.call_args
            assert call_args[0][0] == "parse.request"  # topic
            assert call_args[1]["key"] == "t1"


class TestKafkaConsumer:
    def test_parse_message(self):
        """验证消息反序列化流程。"""
        req = ParseRequest(task_id="t1", file_path="/tmp/a.pdf", filename="a.pdf", kb_id="kb1")
        payload = req.model_dump_json().encode("utf-8")

        parsed = ParseRequest.model_validate_json(payload)
        assert parsed.task_id == "t1"

    def test_process_fn_called(self):
        with patch("src.multidal.queue.consumer.Consumer") as MockConsumer:
            mock_consumer = MagicMock()

            msg = MagicMock()
            msg.error.return_value = None
            req = ParseRequest(task_id="t1", file_path="/tmp/a.pdf", filename="a.pdf", kb_id="kb1")
            msg.value.return_value = req.model_dump_json().encode("utf-8")
            mock_consumer.poll.side_effect = [msg, None]  # first=msg, second=stop signal

            MockConsumer.return_value = mock_consumer

            consumer = KafkaConsumer()
            mock_fn = MagicMock()

            consumer._running = True  # will stop after first message
            # Override poll to return msg then None to trigger stop
            call_count = [0]

            def poll_side_effect(timeout):
                call_count[0] += 1
                if call_count[0] == 1:
                    return msg
                consumer._running = False
                return None

            mock_consumer.poll.side_effect = poll_side_effect
            consumer.start(mock_fn)
            mock_fn.assert_called_once()
