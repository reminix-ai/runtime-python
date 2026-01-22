"""Tests for request/response types."""

import pytest
from pydantic import ValidationError

from reminix_runtime.types import (
    ExecuteRequest,
    ExecuteResponse,
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


class TestExecuteRequest:
    """Tests for ExecuteRequest type."""

    def test_execute_request_defaults_to_empty_input(self):
        """ExecuteRequest defaults to empty input."""
        req = ExecuteRequest()
        assert req.input == {}

    def test_execute_request_accepts_valid_input(self):
        """ExecuteRequest accepts input dict."""
        req = ExecuteRequest(input={"task": "summarize", "text": "hello"})
        assert req.input["task"] == "summarize"

    def test_execute_request_accepts_empty_input(self):
        """ExecuteRequest can have empty input (for chat agents that receive messages)."""
        req = ExecuteRequest(input={})
        assert req.input == {}

    def test_execute_request_accepts_stream(self):
        """ExecuteRequest can have stream flag."""
        req = ExecuteRequest(input={"task": "test"}, stream=True)
        assert req.stream is True

    def test_execute_request_accepts_context(self):
        """ExecuteRequest can have optional context."""
        req = ExecuteRequest(input={"task": "test"}, context={"user_id": "123"})
        assert req.context == {"user_id": "123"}

    def test_execute_request_with_messages(self):
        """ExecuteRequest can have messages in input for chat-style agents."""
        req = ExecuteRequest(
            input={
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        )
        assert len(req.input["messages"]) == 2


class TestExecuteResponse:
    """Tests for ExecuteResponse type.

    ExecuteResponse is now a dict with dynamic keys based on agent's responseKeys.
    - Regular agents: { "output": ... }
    - Chat agents: { "messages": [{ "role": "assistant", "content": "..." }, ...] }
    """

    def test_execute_response_is_dict(self):
        """ExecuteResponse is a dict type alias."""
        resp: ExecuteResponse = {"output": "Result"}
        assert resp["output"] == "Result"

    def test_execute_response_accepts_string_output(self):
        """ExecuteResponse accepts string output."""
        resp: ExecuteResponse = {"output": "Result"}
        assert resp["output"] == "Result"

    def test_execute_response_accepts_dict_output(self):
        """ExecuteResponse accepts dict output."""
        resp: ExecuteResponse = {"output": {"result": 42, "status": "ok"}}
        assert resp["output"] == {"result": 42, "status": "ok"}

    def test_execute_response_accepts_any_output(self):
        """ExecuteResponse accepts any type of output."""
        resp: ExecuteResponse = {"output": [1, 2, 3]}
        assert resp["output"] == [1, 2, 3]

    def test_execute_response_chat_agent_format(self):
        """Chat agents return { messages: [{ role, content }, ...] }."""
        resp: ExecuteResponse = {"messages": [{"role": "assistant", "content": "Hello!"}]}
        assert resp["messages"][0]["role"] == "assistant"
        assert resp["messages"][0]["content"] == "Hello!"
