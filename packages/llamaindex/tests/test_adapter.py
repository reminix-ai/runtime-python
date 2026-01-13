"""Tests for the LlamaIndex adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from reminix_runtime import InvokeRequest, ChatRequest, BaseAdapter
from reminix_llamaindex import wrap, LlamaIndexAdapter


def create_mock_response(content: str = "Hello!"):
    """Create a mock LlamaIndex chat response."""
    mock_response = MagicMock()
    mock_response.response = content
    return mock_response


class TestWrap:
    """Tests for the wrap() function."""

    def test_wrap_returns_adapter(self):
        """wrap() should return a LlamaIndexAdapter."""
        mock_engine = MagicMock()
        adapter = wrap(mock_engine)

        assert isinstance(adapter, LlamaIndexAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_wrap_with_custom_name(self):
        """wrap() should accept a custom name."""
        mock_engine = MagicMock()
        adapter = wrap(mock_engine, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_default_name(self):
        """wrap() should use default name if not provided."""
        mock_engine = MagicMock()
        adapter = wrap(mock_engine)

        assert adapter.name == "llamaindex-agent"


class TestLlamaIndexAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_engine(self):
        """invoke() should call the chat engine."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock(return_value=create_mock_response("Hello!"))

        adapter = wrap(mock_engine)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        mock_engine.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_passes_last_message(self):
        """invoke() should pass the last user message to the engine."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock(return_value=create_mock_response("Response"))

        adapter = wrap(mock_engine)
        request = InvokeRequest(
            messages=[
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "First response"},
                {"role": "user", "content": "Second message"},
            ]
        )

        await adapter.invoke(request)

        # LlamaIndex chat engines typically take the last message
        call_args = mock_engine.achat.call_args
        assert "Second message" in str(call_args)

    @pytest.mark.asyncio
    async def test_invoke_returns_response(self):
        """invoke() should return an InvokeResponse."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock(
            return_value=create_mock_response("Hello from LlamaIndex!")
        )

        adapter = wrap(mock_engine)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.invoke(request)

        assert response.content == "Hello from LlamaIndex!"
        assert len(response.messages) >= 1

    @pytest.mark.asyncio
    async def test_invoke_includes_original_messages(self):
        """invoke() response should include original messages plus response."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock(return_value=create_mock_response("Response"))

        adapter = wrap(mock_engine)
        request = InvokeRequest(messages=[{"role": "user", "content": "Hello"}])

        response = await adapter.invoke(request)

        assert len(response.messages) == 2
        assert response.messages[0]["role"] == "user"
        assert response.messages[0]["content"] == "Hello"
        assert response.messages[1]["role"] == "assistant"
        assert response.messages[1]["content"] == "Response"


class TestLlamaIndexAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_engine(self):
        """chat() should call the chat engine."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock(return_value=create_mock_response("Hello!"))

        adapter = wrap(mock_engine)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_engine.achat.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """chat() should return a ChatResponse."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock(
            return_value=create_mock_response("Chat response")
        )

        adapter = wrap(mock_engine)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.content == "Chat response"
        assert len(response.messages) >= 1
