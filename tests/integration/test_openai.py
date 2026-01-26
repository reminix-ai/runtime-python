"""Integration tests for OpenAI adapter."""

import httpx
import pytest
from httpx import ASGITransport
from openai import AsyncOpenAI

from reminix_openai import wrap_agent
from reminix_runtime import create_app


@pytest.mark.openai
class TestOpenAIAdapter:
    """Integration tests for OpenAI adapter."""

    @pytest.fixture
    def agent(self, openai_api_key):
        client = AsyncOpenAI(api_key=openai_api_key)
        return wrap_agent(client, name="test-openai", model="gpt-4.1-nano")

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
            "/agents/test-openai/execute",
            json={"prompt": "Say 'hello' and nothing else."},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "hello" in data["output"].lower()

    async def test_invoke_with_messages(self, client):
        """Test invoke with messages array."""
        response = await client.post(
            "/agents/test-openai/execute",
            json={"messages": [{"role": "user", "content": "Say 'test' and nothing else."}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert len(data["output"]) > 0

    async def test_chat(self, client):
        """Test chat endpoint."""
        response = await client.post(
            "/agents/test-openai/execute",
            json={"messages": [{"role": "user", "content": "Say 'hi' and nothing else."}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_invoke_stream(self, client):
        """Test streaming invoke endpoint."""
        async with client.stream(
            "POST",
            "/agents/test-openai/execute",
            json={"prompt": "Say 'stream' and nothing else.", "stream": True},
        ) as response:
            assert response.status_code == 200
            chunks = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunks.append(line)
            assert len(chunks) > 0
