"""Tests for the LangChain adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from reminix_langchain import LangChainAdapter, wrap
from reminix_runtime import BaseAdapter, ChatRequest, InvokeRequest


class TestWrap:
    """Tests for the wrap() function."""

    def test_wrap_returns_adapter(self):
        """wrap() should return a LangChainAdapter."""
        mock_runnable = MagicMock(spec=Runnable)
        adapter = wrap(mock_runnable)

        assert isinstance(adapter, LangChainAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_wrap_with_custom_name(self):
        """wrap() should accept a custom name."""
        mock_runnable = MagicMock(spec=Runnable)
        adapter = wrap(mock_runnable, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_default_name(self):
        """wrap() should use default name if not provided."""
        mock_runnable = MagicMock(spec=Runnable)
        adapter = wrap(mock_runnable)

        assert adapter.name == "langchain-agent"


class TestLangChainAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_runnable(self):
        """invoke() should call the underlying runnable with the input."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))

        adapter = wrap(mock_runnable)
        request = InvokeRequest(input={"query": "What is AI?"})

        response = await adapter.invoke(request)

        mock_runnable.ainvoke.assert_called_once_with({"query": "What is AI?"})

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello from LangChain!"))

        adapter = wrap(mock_runnable)
        request = InvokeRequest(input={"query": "Hi"})

        response = await adapter.invoke(request)

        assert response.output == "Hello from LangChain!"

    @pytest.mark.asyncio
    async def test_invoke_handles_dict_response(self):
        """invoke() should handle dict responses from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value={"result": "success", "value": 42})

        adapter = wrap(mock_runnable)
        request = InvokeRequest(input={"task": "compute"})

        response = await adapter.invoke(request)

        assert response.output == {"result": "success", "value": 42}

    @pytest.mark.asyncio
    async def test_invoke_handles_string_response(self):
        """invoke() should handle string responses from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value="Simple string result")

        adapter = wrap(mock_runnable)
        request = InvokeRequest(input={"query": "test"})

        response = await adapter.invoke(request)

        assert response.output == "Simple string result"


class TestLangChainAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_runnable(self):
        """chat() should call the underlying runnable with converted messages."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))

        adapter = wrap(mock_runnable)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_runnable.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_converts_messages_to_langchain_format(self):
        """chat() should convert Reminix messages to LangChain format."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        adapter = wrap(mock_runnable)
        request = ChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ]
        )

        await adapter.chat(request)

        # Check that messages were converted correctly
        call_args = mock_runnable.ainvoke.call_args[0][0]
        assert len(call_args) == 4
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
        assert isinstance(call_args[2], AIMessage)
        assert isinstance(call_args[3], HumanMessage)

    @pytest.mark.asyncio
    async def test_chat_returns_output_and_messages(self):
        """chat() should return output and messages."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Chat response"))

        adapter = wrap(mock_runnable)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.output == "Chat response"
        assert len(response.messages) == 2
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "Chat response"


class TestMessageConversion:
    """Tests for message format conversion."""

    @pytest.mark.asyncio
    async def test_tool_role_handled(self):
        """Tool role messages should be handled appropriately."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        adapter = wrap(mock_runnable)
        request = ChatRequest(
            messages=[
                {"role": "user", "content": "Use a tool"},
                {"role": "tool", "content": "Tool result"},
            ]
        )

        # Should not raise an error
        response = await adapter.chat(request)
        assert response.output == "Response"
