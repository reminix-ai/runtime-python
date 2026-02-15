"""Tests for the LlamaIndex RAG adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_llamaindex import LlamaIndexRag
from reminix_runtime import AGENT_TEMPLATES, AgentRequest


class TestLlamaIndexRag:
    """Tests for the LlamaIndexRag class."""

    def test_instantiation(self):
        """LlamaIndexRag should be instantiable."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRag(mock_engine)

        assert isinstance(agent, LlamaIndexRag)

    def test_custom_name(self):
        """LlamaIndexRag should accept a custom name."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRag(mock_engine, name="my-custom-agent")

        assert agent.name == "my-custom-agent"

    def test_default_name(self):
        """LlamaIndexRag should use default name if not provided."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRag(mock_engine)

        assert agent.name == "llamaindex-agent"

    def test_rag_template_metadata(self):
        """LlamaIndexRag should have rag template metadata."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRag(mock_engine)

        assert agent.metadata["template"] == "rag"
        assert agent.metadata["input"] == AGENT_TEMPLATES["rag"]["input"]


class TestLlamaIndexRagInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_engine(self):
        """invoke() should call the engine with query from input."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRag(mock_engine)
        request = AgentRequest(input={"query": "What is AI?"})

        await agent.invoke(request)

        mock_engine.achat.assert_called_once_with("What is AI?")

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the engine."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRag(mock_engine)
        request = AgentRequest(input={"query": "Hi"})

        response = await agent.invoke(request)

        assert response["output"] == "Hello from LlamaIndex!"

    @pytest.mark.asyncio
    async def test_invoke_with_prompt_input(self):
        """invoke() should handle input with prompt key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRag(mock_engine)
        request = AgentRequest(input={"prompt": "Tell me about AI"})

        await agent.invoke(request)

        mock_engine.achat.assert_called_once_with("Tell me about AI")

    @pytest.mark.asyncio
    async def test_invoke_with_message_input(self):
        """invoke() should handle input with message key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRag(mock_engine)
        request = AgentRequest(input={"message": "Hello there"})

        await agent.invoke(request)

        mock_engine.achat.assert_called_once_with("Hello there")

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should use last user message from messages input."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRag(mock_engine)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "Response 1"},
                    {"role": "user", "content": "Second message"},
                ]
            }
        )

        await agent.invoke(request)

        mock_engine.achat.assert_called_once_with("Second message")
