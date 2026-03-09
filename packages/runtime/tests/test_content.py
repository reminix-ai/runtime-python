"""Tests for content helpers."""

from reminix_runtime.content import build_messages_from_input, message_content_to_text
from reminix_runtime.types import (
    AgentRequest,
    ImageUrlContentPart,
    ImageUrlPart,
    RefusalContentPart,
    TextContentPart,
)


class TestMessageContentToText:
    """Tests for message_content_to_text."""

    def test_string_content(self):
        assert message_content_to_text("hello") == "hello"

    def test_none_content(self):
        assert message_content_to_text(None) == ""

    def test_empty_string(self):
        assert message_content_to_text("") == ""

    def test_text_content_parts(self):
        parts = [TextContentPart(text="hello"), TextContentPart(text="world")]
        assert message_content_to_text(parts) == "hello world"

    def test_mixed_content_parts(self):
        parts = [
            TextContentPart(text="Look at this:"),
            ImageUrlContentPart(image_url=ImageUrlPart(url="https://example.com/img.png")),
            TextContentPart(text="Nice, right?"),
        ]
        assert message_content_to_text(parts) == "Look at this: [image_url] Nice, right?"

    def test_non_text_parts_render_type(self):
        parts = [RefusalContentPart(refusal="I can't do that")]
        assert message_content_to_text(parts) == "[refusal]"


class TestBuildMessagesFromInput:
    """Tests for build_messages_from_input."""

    def test_messages_input(self):
        request = AgentRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        messages = build_messages_from_input(request)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "hi"

    def test_messages_multiple(self):
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "hi"},
                ]
            }
        )
        messages = build_messages_from_input(request)
        assert len(messages) == 2
        assert messages[0].role == "system"
        assert messages[1].role == "user"

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
        assert messages[0].content == '{"key": "value", "num": 42}'

    def test_empty_input(self):
        request = AgentRequest(input={})
        messages = build_messages_from_input(request)
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "{}"
