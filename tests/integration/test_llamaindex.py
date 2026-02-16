"""Integration tests for LlamaIndex agents with tool calling."""

import httpx
import pytest
from httpx import ASGITransport
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from reminix_llamaindex import LlamaIndexRagAgent
from reminix_runtime import create_app


def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Mock weather data
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Unknown weather for {city}")


class ChatEngineWrapper:
    """Wrapper to adapt LlamaIndex ReActAgent workflow to ChatEngine interface."""

    def __init__(self, agent: ReActAgent):
        self._agent = agent
        self._ctx = Context(agent)

    async def achat(self, message: str):
        """Async chat method compatible with LlamaIndex ChatEngine protocol."""
        handler = self._agent.run(message, ctx=self._ctx)
        response = await handler
        return response

    async def astream_chat(self, message: str):
        """Async streaming chat - not implemented."""
        raise NotImplementedError("Streaming not implemented for workflow agents")


@pytest.mark.llamaindex
class TestLlamaIndexAgents:
    """Integration tests for LlamaIndex agents."""

    @pytest.fixture
    def agent(self, openai_api_key):
        llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        react_agent = ReActAgent(tools=[get_weather], llm=llm)
        wrapped_engine = ChatEngineWrapper(react_agent)
        return LlamaIndexRagAgent(wrapped_engine, name="test-llamaindex")

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
            "/agents/test-llamaindex/invoke",
            json={"input": {"query": "Say 'hello' and nothing else."}},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_chat(self, client):
        """Test chat endpoint."""
        response = await client.post(
            "/agents/test-llamaindex/invoke",
            json={
                "input": {"messages": [{"role": "user", "content": "Say 'hi' and nothing else."}]}
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_tool_calling(self, client):
        """Test that the agent calls tools and returns results."""
        response = await client.post(
            "/agents/test-llamaindex/invoke",
            json={
                "input": {"messages": [{"role": "user", "content": "What's the weather in Paris?"}]}
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        # The agent should have called the tool and returned weather info
        output = data["output"].lower()
        assert "sunny" in output or "22" in output or "paris" in output
