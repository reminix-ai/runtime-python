"""Integration tests for OpenAI adapter."""

import pytest
from openai import AsyncOpenAI

from reminix_openai import wrap
from reminix_runtime import InvokeRequest, ChatRequest, Message


@pytest.mark.openai
class TestOpenAIAdapter:
    """Integration tests for OpenAI adapter."""

    @pytest.fixture
    def client(self, openai_api_key):
        return AsyncOpenAI(api_key=openai_api_key)

    @pytest.fixture
    def agent(self, client):
        return wrap(client, name="test-openai", model="gpt-4.1-nano")

    async def test_invoke_with_prompt(self, agent):
        """Test invoke with a simple prompt."""
        request = InvokeRequest(input={"prompt": "Say 'hello' and nothing else."})
        response = await agent.invoke(request)

        assert response.output is not None
        assert len(response.output) > 0
        assert "hello" in response.output.lower()

    async def test_invoke_with_messages(self, agent):
        """Test invoke with messages array."""
        request = InvokeRequest(
            input={
                "messages": [
                    {"role": "user", "content": "Say 'test' and nothing else."}
                ]
            }
        )
        response = await agent.invoke(request)

        assert response.output is not None
        assert len(response.output) > 0

    async def test_chat(self, agent):
        """Test chat with conversation."""
        request = ChatRequest(
            messages=[Message(role="user", content="Say 'hi' and nothing else.")]
        )
        response = await agent.chat(request)

        assert response.output is not None
        assert len(response.output) > 0
        assert len(response.messages) == 2
        assert response.messages[-1]["role"] == "assistant"

    async def test_invoke_stream(self, agent):
        """Test streaming invoke."""
        request = InvokeRequest(input={"prompt": "Say 'stream' and nothing else."})

        chunks = []
        async for chunk in agent.invoke_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0

    async def test_chat_stream(self, agent):
        """Test streaming chat."""
        request = ChatRequest(
            messages=[Message(role="user", content="Say 'ok' and nothing else.")]
        )

        chunks = []
        async for chunk in agent.chat_stream(request):
            chunks.append(chunk)

        assert len(chunks) > 0
