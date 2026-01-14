"""Tests for the Anthropic adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_anthropic import AnthropicAdapter, wrap
from reminix_runtime import BaseAdapter, ChatRequest, InvokeRequest


class TestWrap:
    """Tests for the wrap() function."""

    def test_wrap_returns_adapter(self):
        """wrap() should return an AnthropicAdapter."""
        mock_client = MagicMock()
        adapter = wrap(mock_client)

        assert isinstance(adapter, AnthropicAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_wrap_with_custom_name(self):
        """wrap() should accept a custom name."""
        mock_client = MagicMock()
        adapter = wrap(mock_client, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_with_custom_model(self):
        """wrap() should accept a custom model."""
        mock_client = MagicMock()
        adapter = wrap(mock_client, model="claude-opus-4-20250514")

        assert adapter.model == "claude-opus-4-20250514"

    def test_wrap_default_values(self):
        """wrap() should use default values if not provided."""
        mock_client = MagicMock()
        adapter = wrap(mock_client)

        assert adapter.name == "anthropic-agent"
        assert adapter.model == "claude-sonnet-4-20250514"


class TestAnthropicAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
        request = InvokeRequest(input={"prompt": "Hi"})

        response = await adapter.invoke(request)

        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello from Anthropic!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
        request = InvokeRequest(input={"prompt": "Hi"})

        response = await adapter.invoke(request)

        assert response.output == "Hello from Anthropic!"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should handle input with messages key."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
        request = InvokeRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        response = await adapter.invoke(request)

        assert response.output == "Response"


class TestAnthropicAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_client(self):
        """chat() should call the Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_returns_output_and_messages(self):
        """chat() should return output and messages."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Chat response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.output == "Chat response"
        assert len(response.messages) == 2
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "Chat response"

    @pytest.mark.asyncio
    async def test_chat_extracts_system_message(self):
        """chat() should extract system message for Anthropic API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_client)
        request = ChatRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hi"},
            ]
        )

        await adapter.chat(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"
        # Messages should not include system message
        assert all(m["role"] != "system" for m in call_kwargs["messages"])
