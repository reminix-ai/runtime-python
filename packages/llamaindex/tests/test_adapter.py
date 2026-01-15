"""Tests for the LlamaIndex adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reminix_llamaindex import LlamaIndexAdapter, wrap, wrap_and_serve
from reminix_runtime import BaseAdapter, ChatRequest, InvokeRequest


class TestWrap:
    """Tests for the wrap() function."""

    def test_wrap_returns_adapter(self):
        """wrap() should return a LlamaIndexAdapter."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        adapter = wrap(mock_engine)

        assert isinstance(adapter, LlamaIndexAdapter)
        assert isinstance(adapter, BaseAdapter)

    def test_wrap_with_custom_name(self):
        """wrap() should accept a custom name."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        adapter = wrap(mock_engine, name="my-custom-agent")

        assert adapter.name == "my-custom-agent"

    def test_wrap_default_name(self):
        """wrap() should use default name if not provided."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        adapter = wrap(mock_engine)

        assert adapter.name == "llamaindex-agent"


class TestLlamaIndexAdapterInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_engine(self):
        """invoke() should call the engine with query from input."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_engine)
        request = InvokeRequest(input={"query": "What is AI?"})

        response = await adapter.invoke(request)

        mock_engine.achat.assert_called_once_with("What is AI?")

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the engine."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_engine)
        request = InvokeRequest(input={"query": "Hi"})

        response = await adapter.invoke(request)

        assert response.output == "Hello from LlamaIndex!"

    @pytest.mark.asyncio
    async def test_invoke_with_prompt_input(self):
        """invoke() should handle input with prompt key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_engine)
        request = InvokeRequest(input={"prompt": "Tell me about AI"})

        response = await adapter.invoke(request)

        mock_engine.achat.assert_called_once_with("Tell me about AI")

    @pytest.mark.asyncio
    async def test_invoke_with_message_input(self):
        """invoke() should handle input with message key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_engine)
        request = InvokeRequest(input={"message": "Hello there"})

        response = await adapter.invoke(request)

        mock_engine.achat.assert_called_once_with("Hello there")


class TestLlamaIndexAdapterChat:
    """Tests for the chat() method."""

    @pytest.mark.asyncio
    async def test_chat_calls_engine(self):
        """chat() should call the engine with the last user message."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_engine)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        mock_engine.achat.assert_called_once_with("Hi")

    @pytest.mark.asyncio
    async def test_chat_returns_output_and_messages(self):
        """chat() should return output and messages."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Chat response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_engine)
        request = ChatRequest(messages=[{"role": "user", "content": "Hi"}])

        response = await adapter.chat(request)

        assert response.output == "Chat response"
        assert len(response.messages) == 2
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "Chat response"

    @pytest.mark.asyncio
    async def test_chat_uses_last_user_message(self):
        """chat() should use the last user message in the conversation."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        adapter = wrap(mock_engine)
        request = ChatRequest(
            messages=[
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Second message"},
            ]
        )

        await adapter.chat(request)

        mock_engine.achat.assert_called_once_with("Second message")


class TestWrapAndServe:
    """Tests for the wrap_and_serve() function."""

    def test_wrap_and_serve_is_callable(self):
        """wrap_and_serve() should be callable."""
        assert callable(wrap_and_serve)

    @patch("reminix_llamaindex.adapter.serve")
    def test_wrap_and_serve_calls_serve(self, mock_serve):
        """wrap_and_serve() should call serve with wrapped adapter."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()

        wrap_and_serve(mock_engine, name="test-agent")

        mock_serve.assert_called_once()
        call_args = mock_serve.call_args
        agents = call_args[0][0]
        assert len(agents) == 1
        assert isinstance(agents[0], LlamaIndexAdapter)
        assert agents[0].name == "test-agent"

    @patch("reminix_llamaindex.adapter.serve")
    def test_wrap_and_serve_passes_serve_options(self, mock_serve):
        """wrap_and_serve() should pass port and host to serve."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()

        wrap_and_serve(mock_engine, name="test-agent", port=3000, host="localhost")

        mock_serve.assert_called_once()
        call_kwargs = mock_serve.call_args[1]
        assert call_kwargs["port"] == 3000
        assert call_kwargs["host"] == "localhost"
