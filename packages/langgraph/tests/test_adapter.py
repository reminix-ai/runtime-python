"""Tests for the LangGraph adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from reminix_runtime import InvokeRequest, ChatRequest, BaseAdapter
from reminix_langgraph import wrap, LangGraphAdapter


class TestWrap:
    """Tests for the wrap() function."""

    def test_wrap_returns_adapter(self):
        """wrap() should return a LangGraphAdapter."""
        mock_graph = MagicMock()
        adapter = wrap(mock_graph)

        assert isinstance(adapter, LangGraphAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_wrap_with_custom_name(self):
        """wrap() should accept a custom name."""
        mock_graph = MagicMock()
        adapter = wrap(mock_graph, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_default_name(self):
        """wrap() should use default name if not provided."""
        mock_graph = MagicMock()
        adapter = wrap(mock_graph)

        assert adapter.name == "langgraph-agent"


class TestLangGraphAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_graph(self):
        """invoke() should call the underlying graph."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Hello!")]}
        )

        adapter = wrap(mock_graph)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_passes_messages_in_state(self):
        """invoke() should pass messages in a state dict."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Response")]}
        )

        adapter = wrap(mock_graph)
        request = InvokeRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        )

        await adapter.invoke(request)

        # Check that messages were passed in state dict format
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 2
        assert isinstance(call_args["messages"][0], SystemMessage)
        assert isinstance(call_args["messages"][1], HumanMessage)

    @pytest.mark.asyncio
    async def test_invoke_returns_response(self):
        """invoke() should return an InvokeResponse."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Hello from LangGraph!")]}
        )

        adapter = wrap(mock_graph)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        assert response.content == "Hello from LangGraph!"
        assert len(response.messages) >= 1

    @pytest.mark.asyncio
    async def test_invoke_extracts_last_ai_message(self):
        """invoke() should extract content from the last AI message in response."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="First response"),
                    AIMessage(content="Final response"),
                ]
            }
        )

        adapter = wrap(mock_graph)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hello"}])

        response = await adapter.invoke(request)

        assert response.content == "Final response"

    @pytest.mark.asyncio
    async def test_invoke_includes_full_conversation(self):
        """invoke() response should include full conversation from graph."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                ]
            }
        )

        adapter = wrap(mock_graph)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hello"}])

        response = await adapter.invoke(request)

        assert len(response.messages) == 2
        assert response.messages[0]["role"] == "user"
        assert response.messages[1]["role"] == "assistant"


class TestLangGraphAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_graph(self):
        """chat() should call the underlying graph."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Hello!")]}
        )

        adapter = wrap(mock_graph)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_graph.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """chat() should return a ChatResponse."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={"messages": [AIMessage(content="Chat response")]}
        )

        adapter = wrap(mock_graph)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.content == "Chat response"
        assert len(response.messages) >= 1
