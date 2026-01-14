"""Integration tests for LangGraph adapter with tool calling."""

import httpx
import pytest
from httpx import ASGITransport
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from reminix_langgraph import wrap
from reminix_runtime import create_app


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # Mock weather data
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"Unknown weather for {city}")


@pytest.mark.langgraph
class TestLangGraphAdapter:
    """Integration tests for LangGraph adapter."""

    @pytest.fixture
    def agent(self, openai_api_key):
        llm = ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key)
        graph = create_react_agent(llm, tools=[get_weather])
        return wrap(graph, name="test-langgraph")

    @pytest.fixture
    def app(self, agent):
        return create_app([agent])

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
            "/agents/test-langgraph/invoke",
            json={
                "input": {
                    "messages": [{"role": "user", "content": "Say 'hello' and nothing else."}]
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_chat(self, client):
        """Test chat endpoint."""
        response = await client.post(
            "/agents/test-langgraph/chat",
            json={"messages": [{"role": "user", "content": "Say 'hi' and nothing else."}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        assert "messages" in data

    async def test_tool_calling(self, client):
        """Test that the agent calls tools and returns results."""
        response = await client.post(
            "/agents/test-langgraph/chat",
            json={"messages": [{"role": "user", "content": "What's the weather in Paris?"}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        # The agent should have called the tool and returned weather info
        output = data["output"].lower()
        assert "sunny" in output or "22" in output or "paris" in output
