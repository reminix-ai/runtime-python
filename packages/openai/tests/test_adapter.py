"""Tests for the OpenAI adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_openai import OpenAIAdapter, wrap
from reminix_runtime import BaseAdapter, ChatRequest, InvokeRequest


class TestWrap:
    """Tests for the wrap() function."""

    def test_wrap_returns_adapter(self):
        """wrap() should return an OpenAIAdapter."""
        mock_client = MagicMock()
        adapter = wrap(mock_client)

        assert isinstance(adapter, OpenAIAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_wrap_with_custom_name(self):
        """wrap() should accept a custom name."""
        mock_client = MagicMock()
        adapter = wrap(mock_client, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_with_custom_model(self):
        """wrap() should accept a custom model."""
        mock_client = MagicMock()
        adapter = wrap(mock_client, model="gpt-4o")

        assert adapter.model == "gpt-4o"

    def test_wrap_default_values(self):
        """wrap() should use default values if not provided."""
        mock_client = MagicMock()
        adapter = wrap(mock_client)

        assert adapter.name == "openai-agent"
        assert adapter.model == "gpt-4o-mini"


class TestOpenAIAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
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

        adapter = wrap(mock_client)
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

        adapter = wrap(mock_client)
        request = InvokeRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        response = await adapter.invoke(request)

        # Should pass messages directly
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]


class TestOpenAIAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_client(self):
        """chat() should call the OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
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

        adapter = wrap(mock_client)
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

        adapter = wrap(mock_client, model="gpt-4o")
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        await adapter.chat(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
