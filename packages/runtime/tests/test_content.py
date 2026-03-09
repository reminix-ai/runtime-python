"""Tests for content helpers."""

from reminix_runtime.content import build_messages_from_input, message_content_to_text
from reminix_runtime.types import AgentRequest


class TestMessageContentToText:
    """Tests for message_content_to_text."""

    def test_string_content(self):
        assert message_content_to_text("hello") == "hello"

    def test_none_content(self):
        assert message_content_to_text(None) == ""


class TestBuildMessagesFromInput:
    """Tests for build_messages_from_input."""

    def test_messages_input(self):
        request = AgentRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        messages = build_messages_from_input(request)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "hi"

    def test_prompt_input(self):
        request = AgentRequest(input={"prompt": "hello"})
        messages = build_messages_from_input(request)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "hello"

    def test_fallback_input_produces_valid_json(self):
        request = AgentRequest(input={"key": "value", "num": 42})
        messages = build_messages_from_input(request)
        assert len(messages) == 1
        assert messages[0].role == "user"
        # Should produce valid JSON, not Python repr
        assert messages[0].content == '{"key": "value", "num": 42}'
