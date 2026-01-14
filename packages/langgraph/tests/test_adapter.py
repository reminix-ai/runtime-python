"""Tests for the LangGraph adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from reminix_langgraph import LangGraphAdapter, wrap
from reminix_runtime import BaseAdapter, ChatRequest, InvokeRequest


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
        """invoke() should call the underlying graph with the input."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hello!")]})

        adapter = wrap(mock_graph)
        request = InvokeRequest(input={"query": "What is AI?"})

        response = await adapter.invoke(request)

        mock_graph.ainvoke.assert_called_once_with({"query": "What is AI?"})

    @pytest.mark.asyncio
    async def test_invoke_returns_output_from_messages(self):
        """invoke() should extract output from messages in the result."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
            }
        )

        adapter = wrap(mock_graph)
        request = InvokeRequest(input={"messages": []})

        response = await adapter.invoke(request)

        assert response.output == "Hi there!"

    @pytest.mark.asyncio
    async def test_invoke_handles_dict_result(self):
        """invoke() should handle dict results without messages."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"result": "success"})

        adapter = wrap(mock_graph)
        request = InvokeRequest(input={"task": "compute"})

        response = await adapter.invoke(request)

        assert response.output == {"result": "success"}


class TestLangGraphAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_graph_with_state_dict(self):
        """chat() should call the graph with state dict format."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hello!")]})

        adapter = wrap(mock_graph)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        # Should be called with {"messages": [...]}
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 1
        assert isinstance(call_args["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_chat_returns_output_and_messages(self):
        """chat() should return output and all messages from the graph."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hi"),
                    AIMessage(content="Hello! How can I help?"),
                ]
            }
        )

        adapter = wrap(mock_graph)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.output == "Hello! How can I help?"
        assert len(response.messages) == 2
        assert response.messages[-1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_chat_converts_messages_correctly(self):
        """chat() should convert messages to/from LangChain format."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    SystemMessage(content="You are helpful"),
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi!"),
                ]
            }
        )

        adapter = wrap(mock_graph)
        request = ChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        )

        response = await adapter.chat(request)

        assert response.messages[0]["role"] == "system"
        assert response.messages[1]["role"] == "user"
        assert response.messages[2]["role"] == "assistant"
