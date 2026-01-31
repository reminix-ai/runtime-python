"""Tests for the LangChain adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from reminix_langchain import LangChainAgentAdapter, serve_agent, wrap_agent
from reminix_runtime import AgentAdapter, AgentInvokeRequest


class TestWrap:
    """Tests for the wrap_agent() function."""

    def test_wrap_returns_adapter(self):
        """wrap_agent() should return a LangChainAgentAdapter."""
        mock_runnable = MagicMock(spec=Runnable)
        adapter = wrap_agent(mock_runnable)

        assert isinstance(adapter, LangChainAgentAdapter)
        assert isinstance(adapter, AgentAdapter)

    def test_wrap_with_custom_name(self):
        """wrap_agent() should accept a custom name."""
        mock_runnable = MagicMock(spec=Runnable)
        adapter = wrap_agent(mock_runnable, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_default_name(self):
        """wrap_agent() should use default name if not provided."""
        mock_runnable = MagicMock(spec=Runnable)
        adapter = wrap_agent(mock_runnable)

        assert adapter.name == "langchain-agent"


class TestLangChainAgentAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_runnable(self):
        """invoke() should call the underlying runnable with the input."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))

        adapter = wrap_agent(mock_runnable)
        request = AgentInvokeRequest(input={"query": "What is AI?"})

        response = await adapter.invoke(request)

        mock_runnable.ainvoke.assert_called_once_with({"query": "What is AI?"})

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello from LangChain!"))

        adapter = wrap_agent(mock_runnable)
        request = AgentInvokeRequest(input={"query": "Hi"})

        response = await adapter.invoke(request)

        assert response["output"] == "Hello from LangChain!"

    @pytest.mark.asyncio
    async def test_invoke_handles_dict_response(self):
        """invoke() should handle dict responses from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value={"result": "success", "value": 42})

        adapter = wrap_agent(mock_runnable)
        request = AgentInvokeRequest(input={"task": "compute"})

        response = await adapter.invoke(request)

        assert response["output"] == {"result": "success", "value": 42}

    @pytest.mark.asyncio
    async def test_invoke_handles_string_response(self):
        """invoke() should handle string responses from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value="Simple string result")

        adapter = wrap_agent(mock_runnable)
        request = AgentInvokeRequest(input={"query": "test"})

        response = await adapter.invoke(request)

        assert response["output"] == "Simple string result"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should convert messages to LangChain format."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        adapter = wrap_agent(mock_runnable)
        request = AgentInvokeRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ]
            }
        )

        await adapter.invoke(request)

        # Check that messages were converted correctly
        call_args = mock_runnable.ainvoke.call_args[0][0]
        assert len(call_args) == 4
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
        assert isinstance(call_args[2], AIMessage)
        assert isinstance(call_args[3], HumanMessage)


class TestMessageConversion:
    """Tests for message format conversion."""

    @pytest.mark.asyncio
    async def test_tool_role_handled(self):
        """Tool role messages should be handled appropriately."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        adapter = wrap_agent(mock_runnable)
        request = AgentInvokeRequest(
            input={
                "messages": [
                    {"role": "user", "content": "Use a tool"},
                    {"role": "tool", "content": "Tool result"},
                ]
            }
        )

        # Should not raise an error
        response = await adapter.invoke(request)
        assert response["output"] == "Response"


class TestWrapAndServe:
    """Tests for the serve_agent() function."""

    def test_serve_agent_is_callable(self):
        """serve_agent() should be callable."""
        assert callable(serve_agent)

    @patch("reminix_langchain.agent_adapter.serve")
    def test_serve_agent_calls_serve(self, mock_serve):
        """serve_agent() should call serve with wrapped adapter."""
        mock_runnable = MagicMock(spec=Runnable)

        serve_agent(mock_runnable, name="test-agent")

        mock_serve.assert_called_once()
        call_args = mock_serve.call_args
        agents = call_args.kwargs["agents"]
        assert len(agents) == 1
        assert isinstance(agents[0], LangChainAgentAdapter)
        assert agents[0].name == "test-agent"

    @patch("reminix_langchain.agent_adapter.serve")
    def test_serve_agent_passes_serve_options(self, mock_serve):
        """serve_agent() should pass port and host to serve."""
        mock_runnable = MagicMock(spec=Runnable)

        serve_agent(mock_runnable, name="test-agent", port=3000, host="localhost")

        mock_serve.assert_called_once()
        call_kwargs = mock_serve.call_args[1]
        assert call_kwargs["port"] == 3000
        assert call_kwargs["host"] == "localhost"
