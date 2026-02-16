"""Integration tests for Anthropic agents."""

import httpx
import pytest
from anthropic import AsyncAnthropic
from httpx import ASGITransport

from reminix_anthropic import AnthropicChatAgent
from reminix_runtime import create_app


@pytest.mark.anthropic
class TestAnthropicAgents:
    """Integration tests for Anthropic agents."""

    @pytest.fixture
    def agent(self, anthropic_api_key):
        client = AsyncAnthropic(api_key=anthropic_api_key)
        return AnthropicChatAgent(
            client,
            name="test-anthropic",
            model="claude-sonnet-4-5-20250929",
            max_tokens=100,
        )

    @pytest.fixture
    def app(self, agent):
        return create_app(agents=[agent])

    @pytest.fixture
    async def client(self, app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            yield client

    async def test_invoke(self, client):
        """Test invoke endpoint."""
        response = await client.post(
            "/agents/test-anthropic/invoke",
            json={"input": {"prompt": "Say 'hello' and nothing else."}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "hello" in data["output"].lower()

    async def test_invoke_with_messages(self, client):
        """Test invoke with messages array."""
        response = await client.post(
            "/agents/test-anthropic/invoke",
            json={
                "input": {"messages": [{"role": "user", "content": "Say 'test' and nothing else."}]}
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert len(data["output"]) > 0

    async def test_invoke_with_system_message(self, client):
        """Test invoke with system message."""
        response = await client.post(
            "/agents/test-anthropic/invoke",
            json={
                "input": {
                    "messages": [
                        {"role": "system", "content": "You only respond with 'yes'."},
                        {"role": "user", "content": "Do you understand?"},
                    ]
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "yes" in data["output"].lower()

    async def test_chat(self, client):
        """Test chat endpoint."""
        response = await client.post(
            "/agents/test-anthropic/invoke",
            json={
                "input": {"messages": [{"role": "user", "content": "Say 'hi' and nothing else."}]}
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_invoke_stream(self, client):
        """Test streaming invoke endpoint."""
        async with client.stream(
            "POST",
            "/agents/test-anthropic/invoke",
            json={"input": {"prompt": "Say 'stream' and nothing else."}, "stream": True},
        ) as response:
            assert response.status_code == 200
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
            assert len(chunks) > 0
