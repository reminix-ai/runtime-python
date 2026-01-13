"""Tests for request/response types."""

import pytest
from pydantic import ValidationError

from reminix_runtime.types import (
    Message,
    InvokeRequest,
    InvokeResponse,
    ChatRequest,
    ChatResponse,
)


class TestMessage:
    """Tests for Message type."""

    def test_message_requires_role(self):
        """Message must have a role."""
        with pytest.raises(ValidationError):
            Message(content="hello")  # type: ignore

    def test_message_requires_content(self):
        """Message must have content."""
        with pytest.raises(ValidationError):
            Message(role="user")  # type: ignore

    def test_message_accepts_valid_input(self):
        """Message accepts role and content."""
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_message_role_must_be_valid(self):
        """Message role should be one of: user, assistant, system, tool."""
        # This test will FAIL - we need to add role validation
        valid_roles = ["user", "assistant", "system", "tool"]
        for role in valid_roles:
            msg = Message(role=role, content="test")
            assert msg.role == role

        # Invalid role should raise error
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="test")


class TestInvokeRequest:
    """Tests for InvokeRequest type."""

    def test_invoke_request_requires_messages(self):
        """InvokeRequest must have messages."""
        with pytest.raises(ValidationError):
            InvokeRequest()  # type: ignore

    def test_invoke_request_accepts_valid_input(self):
        """InvokeRequest accepts list of messages."""
        req = InvokeRequest(
            messages=[{"role": "user", "content": "hello"}]
        )
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_invoke_request_messages_cannot_be_empty(self):
        """InvokeRequest must have at least one message."""
        # This test will FAIL - we need to add min_length validation
        with pytest.raises(ValidationError):
            InvokeRequest(messages=[])

    def test_invoke_request_accepts_context(self):
        """InvokeRequest can have optional context."""
        # This test will FAIL - we need to add context field
        req = InvokeRequest(
            messages=[{"role": "user", "content": "hello"}],
            context={"user_id": "123"}
        )
        assert req.context == {"user_id": "123"}


class TestInvokeResponse:
    """Tests for InvokeResponse type."""

    def test_invoke_response_requires_content(self):
        """InvokeResponse must have content."""
        with pytest.raises(ValidationError):
            InvokeResponse(messages=[])  # type: ignore

    def test_invoke_response_requires_messages(self):
        """InvokeResponse must have messages."""
        with pytest.raises(ValidationError):
            InvokeResponse(content="hello")  # type: ignore

    def test_invoke_response_accepts_valid_input(self):
        """InvokeResponse accepts content and messages."""
        resp = InvokeResponse(
            content="Hello!",
            messages=[{"role": "assistant", "content": "Hello!"}]
        )
        assert resp.content == "Hello!"
        assert len(resp.messages) == 1


class TestChatRequest:
    """Tests for ChatRequest type."""

    def test_chat_request_requires_messages(self):
        """ChatRequest must have messages."""
        with pytest.raises(ValidationError):
            ChatRequest()  # type: ignore

    def test_chat_request_accepts_valid_input(self):
        """ChatRequest accepts list of messages."""
        req = ChatRequest(
            messages=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": "how are you?"},
            ]
        )
        assert len(req.messages) == 3

    def test_chat_request_messages_cannot_be_empty(self):
        """ChatRequest must have at least one message."""
        # This test will FAIL - we need to add min_length validation
        with pytest.raises(ValidationError):
            ChatRequest(messages=[])


class TestChatResponse:
    """Tests for ChatResponse type."""

    def test_chat_response_requires_content(self):
        """ChatResponse must have content."""
        with pytest.raises(ValidationError):
            ChatResponse(messages=[])  # type: ignore

    def test_chat_response_requires_messages(self):
        """ChatResponse must have messages."""
        with pytest.raises(ValidationError):
            ChatResponse(content="hello")  # type: ignore

    def test_chat_response_accepts_valid_input(self):
        """ChatResponse accepts content and messages."""
        resp = ChatResponse(
            content="I'm doing well!",
            messages=[
                {"role": "user", "content": "how are you?"},
                {"role": "assistant", "content": "I'm doing well!"},
            ]
        )
        assert resp.content == "I'm doing well!"
        assert len(resp.messages) == 2
