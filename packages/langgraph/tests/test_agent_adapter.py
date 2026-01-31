"""Tests for the LangGraph adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from reminix_langgraph import LangGraphAgentAdapter, serve_agent, wrap_agent
from reminix_runtime import AgentAdapter, AgentInvokeRequest


class TestWrap:
    """Tests for the wrap_agent() function."""

    def test_wrap_returns_adapter(self):
        """wrap_agent() should return a LangGraphAgentAdapter."""
        mock_graph = MagicMock()
        adapter = wrap_agent(mock_graph)

        assert isinstance(adapter, LangGraphAgentAdapter)
        assert isinstance(adapter, AgentAdapter)

    def test_wrap_with_custom_name(self):
        """wrap_agent() should accept a custom name."""
        mock_graph = MagicMock()
        adapter = wrap_agent(mock_graph, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_default_name(self):
        """wrap_agent() should use default name if not provided."""
        mock_graph = MagicMock()
        adapter = wrap_agent(mock_graph)

        assert adapter.name == "langgraph-agent"


class TestLangGraphAgentAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_graph(self):
        """invoke() should call the underlying graph with the input."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hello!")]})

        adapter = wrap_agent(mock_graph)
        request = AgentInvokeRequest(input={"query": "What is AI?"})

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

        adapter = wrap_agent(mock_graph)
        request = AgentInvokeRequest(input={"messages": []})

        response = await adapter.invoke(request)

        assert response["output"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_invoke_handles_dict_result(self):
        """invoke() should handle dict results without messages."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"result": "success"})

        adapter = wrap_agent(mock_graph)
        request = AgentInvokeRequest(input={"task": "compute"})

        response = await adapter.invoke(request)

        assert response["output"] == {"result": "success"}

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should call graph with state dict format for chat-style input."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hello!")]})

        adapter = wrap_agent(mock_graph)
        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        response = await adapter.invoke(request)

        # Should be called with {"messages": [...]}
        call_args = mock_graph.ainvoke.call_args[0][0]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 1
        assert isinstance(call_args["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_invoke_converts_messages_correctly(self):
        """invoke() should convert messages to/from LangChain format."""
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

        adapter = wrap_agent(mock_graph)
        request = AgentInvokeRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        response = await adapter.invoke(request)

        # Output should be extracted from last AI message
        assert response["output"] == "Hi!"


class TestWrapAndServe:
    """Tests for the serve_agent() function."""

    def test_serve_agent_is_callable(self):
        """serve_agent() should be callable."""
        assert callable(serve_agent)

    @patch("reminix_langgraph.agent_adapter.serve")
    def test_serve_agent_calls_serve(self, mock_serve):
        """serve_agent() should call serve with wrapped adapter."""
        mock_graph = MagicMock()

        serve_agent(mock_graph, name="test-agent")

        mock_serve.assert_called_once()
        call_args = mock_serve.call_args
        agents = call_args.kwargs["agents"]
        assert len(agents) == 1
        assert isinstance(agents[0], LangGraphAgentAdapter)
        assert agents[0].name == "test-agent"

    @patch("reminix_langgraph.agent_adapter.serve")
    def test_serve_agent_passes_serve_options(self, mock_serve):
        """serve_agent() should pass port and host to serve."""
        mock_graph = MagicMock()

        serve_agent(mock_graph, name="test-agent", port=3000, host="localhost")

        mock_serve.assert_called_once()
        call_kwargs = mock_serve.call_args[1]
        assert call_kwargs["port"] == 3000
        assert call_kwargs["host"] == "localhost"
