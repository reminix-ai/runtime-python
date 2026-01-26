"""Integration tests for LangChain adapter with tool calling."""

import httpx
import pytest
from httpx import ASGITransport
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from reminix_langchain import wrap_agent
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


@pytest.mark.langchain
class TestLangChainAdapter:
    """Integration tests for LangChain adapter."""

    @pytest.fixture
    def agent(self, openai_api_key):
        llm = ChatOpenAI(model="gpt-4.1-nano", api_key=openai_api_key)
        llm_with_tools = llm.bind_tools([get_weather])
        return wrap_agent(llm_with_tools, name="test-langchain")

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
            "/agents/test-langchain/execute",
            json={"messages": [{"role": "user", "content": "Say 'hello' and nothing else."}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_chat(self, client):
        """Test chat endpoint."""
        response = await client.post(
            "/agents/test-langchain/execute",
            json={"messages": [{"role": "user", "content": "Say 'hi' and nothing else."}]},
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data

    async def test_tool_calling(self, client):
        """Test that the model can call tools."""
        response = await client.post(
            "/agents/test-langchain/execute",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "What's the weather in Paris? Use the get_weather tool.",
                    }
                ]
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "output" in data
        # The model should either call the tool or mention weather
        # Note: With bind_tools, the model returns tool_calls but doesn't execute them
        # This test verifies the integration works
