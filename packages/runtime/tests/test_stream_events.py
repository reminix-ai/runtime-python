"""Tests for stream event types and normalization."""

from reminix_runtime.server import normalize_stream_chunk
from reminix_runtime.stream_events import (
    MessageEvent,
    PendingAction,
    StepEvent,
    TextDeltaEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from reminix_runtime.types import Message, ToolCall


class TestNormalizeStreamChunk:
    """Tests for normalize_stream_chunk()."""

    def test_wraps_string_as_text_delta(self):
        result = normalize_stream_chunk("hello")
        assert result == {"type": "text_delta", "delta": "hello"}

    def test_wraps_empty_string_as_text_delta(self):
        result = normalize_stream_chunk("")
        assert result == {"type": "text_delta", "delta": ""}

    def test_serializes_text_delta_event(self):
        event = TextDeltaEvent(delta="world")
        result = normalize_stream_chunk(event)
        assert result == {"type": "text_delta", "delta": "world"}

    def test_serializes_tool_call_event(self):
        event = ToolCallEvent(
            tool_call=ToolCall(
                id="call_1",
                type="function",
                function={"name": "get_weather", "arguments": '{"city":"Paris"}'},
            )
        )
        result = normalize_stream_chunk(event)
        assert result["type"] == "tool_call"
        assert result["tool_call"]["id"] == "call_1"

    def test_serializes_tool_result_event(self):
        event = ToolResultEvent(tool_call_id="call_1", output="Sunny, 22°C")
        result = normalize_stream_chunk(event)
        assert result == {
            "type": "tool_result",
            "tool_call_id": "call_1",
            "output": "Sunny, 22°C",
        }

    def test_serializes_message_event(self):
        event = MessageEvent(message=Message(role="assistant", content="Hello!"))
        result = normalize_stream_chunk(event)
        assert result["type"] == "message"
        assert result["message"]["role"] == "assistant"
        assert result["message"]["content"] == "Hello!"

    def test_serializes_step_event(self):
        event = StepEvent(name="fetch_data", status="completed", output={"records": 10})
        result = normalize_stream_chunk(event)
        assert result["type"] == "step"
        assert result["name"] == "fetch_data"
        assert result["status"] == "completed"
        assert result["output"] == {"records": 10}

    def test_step_event_excludes_none_fields(self):
        event = StepEvent(name="node1", status="completed")
        result = normalize_stream_chunk(event)
        assert "output" not in result
        assert "pendingAction" not in result

    def test_step_event_with_pending_action_includes_all_fields(self):
        action = PendingAction(
            step="review",
            type="approval",
            message="Approve deployment?",
            options=["approve", "reject"],
            inputSchema={"type": "object", "properties": {"reason": {"type": "string"}}},
            assignee="admin@example.com",
        )
        event = StepEvent(name="deploy", status="paused", pendingAction=action)
        result = normalize_stream_chunk(event)
        pa = result["pendingAction"]
        assert pa["step"] == "review"
        assert pa["type"] == "approval"
        assert pa["message"] == "Approve deployment?"
        assert pa["options"] == ["approve", "reject"]
        assert pa["inputSchema"]["type"] == "object"
        assert pa["assignee"] == "admin@example.com"
