"""Tests for request/response types."""

import pytest
from pydantic import ValidationError

from reminix_runtime.types import (
    InvokeRequest,
    InvokeResponse,
    Message,
)


class TestMessage:
    """Tests for Message type."""

    def test_message_requires_role(self):
        """Message must have a role."""
        with pytest.raises(ValidationError):
            Message(content="hello")  # type: ignore

    def test_message_accepts_valid_input(self):
        """Message accepts role and content."""
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_content_can_be_none(self):
        """Message content can be None (for tool_calls)."""
        msg = Message(
            role="assistant",
            content=None,
            tool_calls=[
                {"id": "1", "type": "function", "function": {"name": "test", "arguments": "{}"}}
            ],
        )
        assert msg.content is None
        assert msg.tool_calls is not None

    def test_message_role_must_be_valid(self):
        """Message role should be one of: user, assistant, system, tool."""
        valid_roles = ["user", "assistant", "system", "tool"]
        for role in valid_roles:
            msg = Message(role=role, content="test")
            assert msg.role == role

        # Invalid role should raise error
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="test")

    def test_message_tool_fields(self):
        """Message can have tool_call_id and name for tool messages."""
        msg = Message(role="tool", content="result", tool_call_id="call_123", name="my_tool")
        assert msg.tool_call_id == "call_123"
        assert msg.name == "my_tool"


class TestInvokeRequest:
    """Tests for InvokeRequest type."""

    def test_invoke_request_defaults_to_empty_input(self):
        """InvokeRequest defaults to empty input."""
        req = InvokeRequest()
        assert req.input == {}

    def test_invoke_request_accepts_valid_input(self):
        """InvokeRequest accepts input dict."""
        req = InvokeRequest(input={"task": "summarize", "text": "hello"})
        assert req.input["task"] == "summarize"

    def test_invoke_request_accepts_empty_input(self):
        """InvokeRequest can have empty input (for chat agents that receive messages)."""
        req = InvokeRequest(input={})
        assert req.input == {}

    def test_invoke_request_accepts_stream(self):
        """InvokeRequest can have stream flag."""
        req = InvokeRequest(input={"task": "test"}, stream=True)
        assert req.stream is True

    def test_invoke_request_accepts_context(self):
        """InvokeRequest can have optional context."""
        req = InvokeRequest(input={"task": "test"}, context={"user_id": "123"})
        assert req.context == {"user_id": "123"}

    def test_invoke_request_with_messages(self):
        """InvokeRequest can have messages in input for chat-style agents."""
        req = InvokeRequest(
            input={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        )
        assert len(req.input["messages"]) == 2


class TestInvokeResponse:
    """Tests for InvokeResponse type.

    InvokeResponse now uses { output: ..., metadata?: ... } format.
    """

    def test_invoke_response_is_dict(self):
        """InvokeResponse is a dict type alias."""
        resp: InvokeResponse = {"output": "Result"}
        assert resp["output"] == "Result"

    def test_invoke_response_accepts_string_output(self):
        """InvokeResponse accepts string output."""
        resp: InvokeResponse = {"output": "Result"}
        assert resp["output"] == "Result"

    def test_invoke_response_accepts_dict_output(self):
        """InvokeResponse accepts dict output."""
        resp: InvokeResponse = {"output": {"result": 42, "status": "ok"}}
        assert resp["output"] == {"result": 42, "status": "ok"}

    def test_invoke_response_accepts_any_output(self):
        """InvokeResponse accepts any type of output."""
        resp: InvokeResponse = {"output": [1, 2, 3]}
        assert resp["output"] == [1, 2, 3]

    def test_invoke_response_accepts_metadata(self):
        """InvokeResponse accepts optional metadata."""
        resp: InvokeResponse = {
            "output": "Result",
            "metadata": {"model": "gpt-4", "latency_ms": 100},
        }
        assert resp["metadata"]["model"] == "gpt-4"
        assert resp["metadata"]["latency_ms"] == 100
