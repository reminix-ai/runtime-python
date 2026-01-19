"""Tests for the OpenAI agent adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reminix_openai import OpenAIAgentAdapter, serve_agent, wrap_agent
from reminix_runtime import AgentAdapter, ChatRequest, InvokeRequest


class TestWrap:
    """Tests for the wrap_agent() function."""

    def test_wrap_returns_adapter(self):
        """wrap_agent() should return an OpenAIAgentAdapter."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client)

        assert isinstance(adapter, OpenAIAgentAdapter)
        assert isinstance(adapter, AgentAdapter)

    def test_wrap_with_custom_name(self):
        """wrap_agent() should accept a custom name."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_with_custom_model(self):
        """wrap_agent() should accept a custom model."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client, model="gpt-4o")

        assert adapter.model == "gpt-4o"

    def test_wrap_default_values(self):
        """wrap_agent() should use default values if not provided."""
        mock_client = MagicMock()
        adapter = wrap_agent(mock_client)

        assert adapter.name == "openai-agent"
        assert adapter.model == "gpt-4o-mini"


class TestOpenAIAgentAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = InvokeRequest(input={"prompt": "Hi"})

        response = await adapter.invoke(request)

        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = InvokeRequest(input={"prompt": "Hi"})

        response = await adapter.invoke(request)

        assert response.output == "Hello from OpenAI!"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should handle input with messages key."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = InvokeRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        response = await adapter.invoke(request)

        # Should pass messages directly
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


class TestOpenAIAgentAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_client(self):
        """chat() should call the OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_returns_output_and_messages(self):
        """chat() should return output and messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Chat response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.output == "Chat response"
        assert len(response.messages) == 2
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "Chat response"

    @pytest.mark.asyncio
    async def test_chat_passes_model(self):
        """chat() should use the configured model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap_agent(mock_client, model="gpt-4o")
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        await adapter.chat(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"


class TestWrapAndServe:
    """Tests for the serve_agent() function."""

    def test_serve_agent_is_callable(self):
        """serve_agent() should be callable."""
        assert callable(serve_agent)

    @patch("reminix_openai.agent_adapter.serve")
    def test_serve_agent_calls_serve(self, mock_serve):
        """serve_agent() should call serve with wrapped adapter."""
        mock_client = MagicMock()

        serve_agent(mock_client, name="test-agent")

        mock_serve.assert_called_once()
        call_args = mock_serve.call_args
        agents = call_args.kwargs["agents"]
        assert len(agents) == 1
        assert isinstance(agents[0], OpenAIAgentAdapter)
        assert agents[0].name == "test-agent"

    @patch("reminix_openai.agent_adapter.serve")
    def test_serve_agent_passes_serve_options(self, mock_serve):
        """serve_agent() should pass port and host to serve."""
        mock_client = MagicMock()

        serve_agent(mock_client, name="test-agent", port=3000, host="localhost")

        mock_serve.assert_called_once()
        call_kwargs = mock_serve.call_args[1]
        assert call_kwargs["port"] == 3000
        assert call_kwargs["host"] == "localhost"
