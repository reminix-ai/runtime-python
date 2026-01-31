"""Tests for the LlamaIndex adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reminix_llamaindex import LlamaIndexAgentAdapter, serve_agent, wrap_agent
from reminix_runtime import AgentAdapter, AgentInvokeRequest


class TestWrap:
    """Tests for the wrap_agent() function."""

    def test_wrap_returns_adapter(self):
        """wrap_agent() should return a LlamaIndexAgentAdapter."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        adapter = wrap_agent(mock_engine)

        assert isinstance(adapter, LlamaIndexAgentAdapter)
        assert isinstance(adapter, AgentAdapter)

    def test_wrap_with_custom_name(self):
        """wrap_agent() should accept a custom name."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        adapter = wrap_agent(mock_engine, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_default_name(self):
        """wrap_agent() should use default name if not provided."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        adapter = wrap_agent(mock_engine)

        assert adapter.name == "llamaindex-agent"


class TestLlamaIndexAgentAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_engine(self):
        """invoke() should call the engine with query from input."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_engine)
        request = AgentInvokeRequest(input={"query": "What is AI?"})

        response = await adapter.invoke(request)

        mock_engine.achat.assert_called_once_with("What is AI?")

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the engine."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_engine)
        request = AgentInvokeRequest(input={"query": "Hi"})

        response = await adapter.invoke(request)

        assert response["output"] == "Hello from LlamaIndex!"

    @pytest.mark.asyncio
    async def test_invoke_with_prompt_input(self):
        """invoke() should handle input with prompt key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_engine)
        request = AgentInvokeRequest(input={"prompt": "Tell me about AI"})

        response = await adapter.invoke(request)

        mock_engine.achat.assert_called_once_with("Tell me about AI")

    @pytest.mark.asyncio
    async def test_invoke_with_message_input(self):
        """invoke() should handle input with message key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_engine)
        request = AgentInvokeRequest(input={"message": "Hello there"})

        response = await adapter.invoke(request)

        mock_engine.achat.assert_called_once_with("Hello there")

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should use last user message from messages input."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_engine)
        request = AgentInvokeRequest(
            input={
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "Response 1"},
                    {"role": "user", "content": "Second message"},
                ]
            }
        )

        await adapter.invoke(request)

        mock_engine.achat.assert_called_once_with("Second message")


class TestWrapAndServe:
    """Tests for the serve_agent() function."""

    def test_serve_agent_is_callable(self):
        """serve_agent() should be callable."""
        assert callable(serve_agent)

    @patch("reminix_llamaindex.agent_adapter.serve")
    def test_serve_agent_calls_serve(self, mock_serve):
        """serve_agent() should call serve with wrapped adapter."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()

        serve_agent(mock_engine, name="test-agent")

        mock_serve.assert_called_once()
        call_args = mock_serve.call_args
        agents = call_args.kwargs["agents"]
        assert len(agents) == 1
        assert isinstance(agents[0], LlamaIndexAgentAdapter)
        assert agents[0].name == "test-agent"

    @patch("reminix_llamaindex.agent_adapter.serve")
    def test_serve_agent_passes_serve_options(self, mock_serve):
        """serve_agent() should pass port and host to serve."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()

        serve_agent(mock_engine, name="test-agent", port=3000, host="localhost")

        mock_serve.assert_called_once()
        call_kwargs = mock_serve.call_args[1]
        assert call_kwargs["port"] == 3000
        assert call_kwargs["host"] == "localhost"
