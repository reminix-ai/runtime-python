"""Tests for the LangChain adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import Runnable

from reminix_runtime import InvokeRequest, ChatRequest, BaseAdapter
from reminix_langchain import wrap, LangChainAdapter


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
        """invoke() should call the underlying runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))

        adapter = wrap(mock_runnable)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        mock_runnable.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_converts_messages_to_langchain_format(self):
        """invoke() should convert Reminix messages to LangChain format."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        adapter = wrap(mock_runnable)
        request = InvokeRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
                {"role": "user", "content": "How are you?"},
            ]
        )

        await adapter.invoke(request)

        # Check that messages were converted correctly
        call_args = mock_runnable.ainvoke.call_args[0][0]
        assert len(call_args) == 4
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
        assert isinstance(call_args[2], AIMessage)
        assert isinstance(call_args[3], HumanMessage)

    @pytest.mark.asyncio
    async def test_invoke_returns_response(self):
        """invoke() should return an InvokeResponse."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello from LangChain!"))

        adapter = wrap(mock_runnable)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        assert response.content == "Hello from LangChain!"
        assert len(response.messages) >= 1
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "Hello from LangChain!"

    @pytest.mark.asyncio
    async def test_invoke_includes_original_messages(self):
        """invoke() response should include the original messages plus the response."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        adapter = wrap(mock_runnable)
        request = InvokeRequest(
            messages=[
                {"role": "user", "content": "Hello"},
            ]
        )

        response = await adapter.invoke(request)

        # Should include original message + response
        assert len(response.messages) == 2
        assert response.messages[0]["role"] == "user"
        assert response.messages[0]["content"] == "Hello"
        assert response.messages[1]["role"] == "assistant"
        assert response.messages[1]["content"] == "Response"


class TestLangChainAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_runnable(self):
        """chat() should call the underlying runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))

        adapter = wrap(mock_runnable)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_runnable.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """chat() should return a ChatResponse."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Chat response"))

        adapter = wrap(mock_runnable)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.content == "Chat response"
        assert len(response.messages) >= 1


class TestMessageConversion:
    """Tests for message format conversion."""

    @pytest.mark.asyncio
    async def test_tool_role_handled(self):
        """Tool role messages should be handled appropriately."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        adapter = wrap(mock_runnable)
        request = InvokeRequest(
            messages=[
                {"role": "user", "content": "Use a tool"},
                {"role": "tool", "content": "Tool result"},
            ]
        )

        # Should not raise an error
        response = await adapter.invoke(request)
        assert response.content == "Response"
