"""Integration tests for Anthropic adapter."""

import pytest
from anthropic import AsyncAnthropic

from reminix_anthropic import wrap
from reminix_runtime import InvokeRequest, ChatRequest, Message


@pytest.mark.anthropic
class TestAnthropicAdapter:
    """Integration tests for Anthropic adapter."""

    @pytest.fixture
    def client(self, anthropic_api_key):
        return AsyncAnthropic(api_key=anthropic_api_key)

    @pytest.fixture
    def agent(self, client):
        return wrap(
            client,
            name="test-anthropic",
            model="claude-3-haiku-20240307",
            max_tokens=100,
        )

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

    async def test_invoke_with_system_message(self, agent):
        """Test invoke with system message."""
        request = InvokeRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You only respond with 'yes'."},
                    {"role": "user", "content": "Do you understand?"},
                ]
            }
        )
        response = await agent.invoke(request)

        assert response.output is not None
        assert "yes" in response.output.lower()

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
