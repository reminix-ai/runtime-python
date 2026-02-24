"""Tests for the LlamaIndex RAG agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_llamaindex import LlamaIndexRagAgent
from reminix_runtime import AGENT_TYPES, AgentRequest


class TestLlamaIndexRagAgent:
    """Tests for the LlamaIndexRagAgent class."""

    def test_instantiation_with_chat_engine(self):
        """LlamaIndexRagAgent should be instantiable with a chat engine."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRagAgent(mock_engine)

        assert isinstance(agent, LlamaIndexRagAgent)

    def test_instantiation_with_query_engine(self):
        """LlamaIndexRagAgent should be instantiable with a query engine."""
        mock_engine = MagicMock(spec=[])
        mock_engine.aquery = AsyncMock()
        agent = LlamaIndexRagAgent(mock_engine)

        assert isinstance(agent, LlamaIndexRagAgent)

    def test_custom_name(self):
        """LlamaIndexRagAgent should accept a custom name."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRagAgent(mock_engine, name="my-custom-agent")

        assert agent.name == "my-custom-agent"

    def test_default_name(self):
        """LlamaIndexRagAgent should use default name if not provided."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRagAgent(mock_engine)

        assert agent.name == "llamaindex-agent"

    def test_rag_type_metadata(self):
        """LlamaIndexRagAgent should have rag type metadata."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRagAgent(mock_engine)

        assert agent.metadata["type"] == "rag"
        assert agent.metadata["inputSchema"] == AGENT_TYPES["rag"]["inputSchema"]

    def test_chat_engine_streaming_enabled(self):
        """Chat engine agents should have streaming enabled."""
        mock_engine = MagicMock()
        mock_engine.achat = AsyncMock()
        agent = LlamaIndexRagAgent(mock_engine)

        assert agent.metadata["capabilities"]["streaming"] is True

    def test_query_engine_streaming_disabled(self):
        """Query engine agents should have streaming disabled."""
        mock_engine = MagicMock(spec=[])
        mock_engine.aquery = AsyncMock()
        agent = LlamaIndexRagAgent(mock_engine)

        assert agent.metadata["capabilities"]["streaming"] is False


class TestLlamaIndexRagAgentChatEngine:
    """Tests for invoke() with a chat engine."""

    @pytest.mark.asyncio
    async def test_invoke_calls_engine(self):
        """invoke() should call achat with query from input."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(input={"query": "What is AI?"})

        await agent.invoke(request)

        mock_engine.achat.assert_called_once_with("What is AI?")

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the engine."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Hello from LlamaIndex!")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(input={"query": "Hi"})

        response = await agent.invoke(request)

        assert response["output"] == "Hello from LlamaIndex!"

    @pytest.mark.asyncio
    async def test_invoke_with_prompt_input(self):
        """invoke() should handle input with prompt key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(input={"prompt": "Tell me about AI"})

        await agent.invoke(request)

        mock_engine.achat.assert_called_once_with("Tell me about AI")

    @pytest.mark.asyncio
    async def test_invoke_with_message_input(self):
        """invoke() should handle input with message key."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(input={"message": "Hello there"})

        await agent.invoke(request)

        mock_engine.achat.assert_called_once_with("Hello there")

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should use last user message from messages input."""
        mock_engine = MagicMock()
        mock_response = MagicMock(response="Response")
        mock_engine.achat = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
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


class TestLlamaIndexRagAgentQueryEngine:
    """Tests for invoke() with a query engine."""

    @pytest.mark.asyncio
    async def test_invoke_calls_aquery(self):
        """invoke() should call aquery with query from input."""
        mock_engine = MagicMock(spec=[])
        mock_response = MagicMock(response="Query result")
        mock_engine.aquery = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(input={"query": "What is AI?"})

        await agent.invoke(request)

        mock_engine.aquery.assert_called_once_with("What is AI?")

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the query engine."""
        mock_engine = MagicMock(spec=[])
        mock_response = MagicMock(response="Query result")
        mock_engine.aquery = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(input={"query": "Hi"})

        response = await agent.invoke(request)

        assert response["output"] == "Query result"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should extract last user message for query engines too."""
        mock_engine = MagicMock(spec=[])
        mock_response = MagicMock(response="Result")
        mock_engine.aquery = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "user", "content": "First"},
                    {"role": "user", "content": "Second"},
                ]
            }
        )

        await agent.invoke(request)

        mock_engine.aquery.assert_called_once_with("Second")

    @pytest.mark.asyncio
    async def test_invoke_stream_falls_back_to_full_response(self):
        """invoke_stream() should yield full response for query engines."""
        mock_engine = MagicMock(spec=[])
        mock_response = MagicMock(response="Full query result")
        mock_engine.aquery = AsyncMock(return_value=mock_response)

        agent = LlamaIndexRagAgent(mock_engine)
        request = AgentRequest(input={"query": "What is AI?"})

        chunks = []
        async for chunk in agent.invoke_stream(request):
            chunks.append(chunk)

        assert chunks == ["Full query result"]
