"""Tests for the Anthropic adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from reminix_runtime import InvokeRequest, ChatRequest, BaseAdapter
from reminix_anthropic import wrap, AnthropicAdapter


def create_mock_response(content: str = "Hello!"):
    """Create a mock Anthropic message response."""
    mock_content_block = MagicMock()
    mock_content_block.type = "text"
    mock_content_block.text = content

    mock_response = MagicMock()
    mock_response.content = [mock_content_block]
    mock_response.role = "assistant"

    return mock_response


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

    def test_wrap_default_name(self):
        """wrap() should use default name if not provided."""
        mock_client = MagicMock()
        adapter = wrap(mock_client)

        assert adapter.name == "anthropic-agent"

    def test_wrap_with_model(self):
        """wrap() should accept a model parameter."""
        mock_client = MagicMock()
        adapter = wrap(mock_client, model="claude-sonnet-4-20250514")

        assert adapter.model == "claude-sonnet-4-20250514"


class TestAnthropicAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the Anthropic client."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=create_mock_response("Hello!")
        )

        adapter = wrap(mock_client)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_passes_messages(self):
        """invoke() should pass messages to the client."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=create_mock_response("Response")
        )

        adapter = wrap(mock_client, model="claude-sonnet-4-20250514")
        request = InvokeRequest(
            messages=[
                {"role": "user", "content": "Hello"},
            ]
        )

        await adapter.invoke(request)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert call_kwargs["model"] == "claude-sonnet-4-20250514"
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_invoke_extracts_system_message(self):
        """invoke() should extract system messages and pass separately."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=create_mock_response("Response")
        )

        adapter = wrap(mock_client)
        request = InvokeRequest(
            messages=[
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ]
        )

        await adapter.invoke(request)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        # System message should be passed as 'system' parameter
        assert call_kwargs["system"] == "You are helpful"
        # Only user message should be in messages
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_invoke_returns_response(self):
        """invoke() should return an InvokeResponse."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=create_mock_response("Hello from Anthropic!")
        )

        adapter = wrap(mock_client)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        assert response.content == "Hello from Anthropic!"
        assert len(response.messages) >= 1

    @pytest.mark.asyncio
    async def test_invoke_includes_original_messages(self):
        """invoke() response should include original messages plus response."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=create_mock_response("Response")
        )

        adapter = wrap(mock_client)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hello"}])

        response = await adapter.invoke(request)

        assert len(response.messages) == 2
        assert response.messages[0]["role"] == "user"
        assert response.messages[0]["content"] == "Hello"
        assert response.messages[1]["role"] == "assistant"
        assert response.messages[1]["content"] == "Response"


class TestAnthropicAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_client(self):
        """chat() should call the Anthropic client."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=create_mock_response("Hello!")
        )

        adapter = wrap(mock_client)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """chat() should return a ChatResponse."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=create_mock_response("Chat response")
        )

        adapter = wrap(mock_client)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.content == "Chat response"
        assert len(response.messages) >= 1
