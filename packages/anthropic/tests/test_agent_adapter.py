"""Tests for the Anthropic adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reminix_anthropic import AnthropicAgentAdapter, serve_agent, wrap_agent
from reminix_runtime import AgentAdapter, AgentInvokeRequest


class TestWrap:
    """Tests for the wrap_agent() function."""

    def test_wrap_returns_adapter(self):
        """wrap_agent() should return an AnthropicAgentAdapter."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client)

        assert isinstance(adapter, AnthropicAgentAdapter)
        assert isinstance(adapter, AgentAdapter)

    def test_wrap_with_custom_name(self):
        """wrap_agent() should accept a custom name."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_with_custom_model(self):
        """wrap_agent() should accept a custom model."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client, model="claude-opus-4-20250514")

        assert adapter.model == "claude-opus-4-20250514"

    def test_wrap_default_values(self):
        """wrap_agent() should use default values if not provided."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client)

        assert adapter.name == "anthropic-agent"
        assert adapter.model == "claude-sonnet-4-20250514"


class TestAnthropicAgentAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = AgentInvokeRequest(input={"prompt": "Hi"})

        response = await adapter.invoke(request)

        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello from Anthropic!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = AgentInvokeRequest(input={"prompt": "Hi"})

        response = await adapter.invoke(request)

        assert response["output"] == "Hello from Anthropic!"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should handle input with messages key."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        response = await adapter.invoke(request)

        assert response["output"] == "Response"

    @pytest.mark.asyncio
    async def test_invoke_extracts_system_message(self):
        """invoke() should extract system message for Anthropic API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = AgentInvokeRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ]
            }
        )

        await adapter.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"
        # Messages should not include system message
        assert all(m["role"] != "system" for m in call_kwargs["messages"])


class TestWrapAndServe:
    """Tests for the serve_agent() function."""

    def test_serve_agent_is_callable(self):
        """serve_agent() should be callable."""
        assert callable(serve_agent)

    @patch("reminix_anthropic.agent_adapter.serve")
    def test_serve_agent_calls_serve(self, mock_serve):
        """serve_agent() should call serve with wrapped adapter."""
        mock_client = MagicMock()

        serve_agent(mock_client, name="test-agent")

        mock_serve.assert_called_once()
        call_args = mock_serve.call_args
        agents = call_args.kwargs["agents"]
        assert len(agents) == 1
        assert isinstance(agents[0], AnthropicAgentAdapter)
        assert agents[0].name == "test-agent"

    @patch("reminix_anthropic.agent_adapter.serve")
    def test_serve_agent_passes_serve_options(self, mock_serve):
        """serve_agent() should pass port and host to serve."""
        mock_client = MagicMock()

        serve_agent(mock_client, name="test-agent", port=3000, host="localhost")

        mock_serve.assert_called_once()
        call_kwargs = mock_serve.call_args[1]
        assert call_kwargs["port"] == 3000
        assert call_kwargs["host"] == "localhost"
