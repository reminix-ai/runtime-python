"""Tests for message conversion utilities."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from reminix_langchain import from_langchain_message, to_langchain_message
from reminix_runtime import Message


class TestToLangChainMessage:
    """Tests for to_langchain_message."""

    def test_user_to_human(self):
        msg = Message(role="user", content="Hello")
        result = to_langchain_message(msg)
        assert isinstance(result, HumanMessage)
        assert result.content == "Hello"

    def test_assistant_to_ai(self):
        msg = Message(role="assistant", content="Hi there")
        result = to_langchain_message(msg)
        assert isinstance(result, AIMessage)
        assert result.content == "Hi there"

    def test_system_to_system(self):
        msg = Message(role="system", content="You are helpful")
        result = to_langchain_message(msg)
        assert isinstance(result, SystemMessage)
        assert result.content == "You are helpful"

    def test_tool_to_tool(self):
        msg = Message(role="tool", content="Tool output", tool_call_id="call_1")
        result = to_langchain_message(msg)
        assert isinstance(result, ToolMessage)
        assert result.content == "Tool output"


class TestFromLangChainMessage:
    """Tests for from_langchain_message."""

    def test_human_to_user(self):
        msg = HumanMessage(content="Hello")
        result = from_langchain_message(msg)
        assert result["role"] == "user"
        assert result["content"] == "Hello"

    def test_ai_to_assistant(self):
        msg = AIMessage(content="Hi")
        result = from_langchain_message(msg)
        assert result["role"] == "assistant"
        assert result["content"] == "Hi"

    def test_system_to_system(self):
        msg = SystemMessage(content="System prompt")
        result = from_langchain_message(msg)
        assert result["role"] == "system"
        assert result["content"] == "System prompt"

    def test_tool_to_tool(self):
        msg = ToolMessage(content="Result", tool_call_id="call_1")
        result = from_langchain_message(msg)
        assert result["role"] == "tool"
        assert result["content"] == "Result"
        assert result["tool_call_id"] == "call_1"

    def test_ai_with_tool_calls(self):
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "call_1", "name": "search", "args": {"q": "test"}}],
        )
        result = from_langchain_message(msg)
        assert result["role"] == "assistant"
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "search"
        assert json.loads(result["tool_calls"][0]["function"]["arguments"]) == {"q": "test"}
