"""Tests for the Google Gemini thread agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_google import GoogleThreadAgent
from reminix_runtime import AGENT_TYPES, AgentRequest


def make_mock_tool(name: str = "get_weather", result: dict | None = None):
    """Create a mock Tool."""
    tool = MagicMock()
    tool.name = name
    tool.metadata = {
        "description": f"Mock {name} tool",
        "input": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    }
    tool.call = AsyncMock(return_value={"output": result or {"temp": 22, "condition": "sunny"}})
    return tool


def make_text_part(text: str = "Hello!"):
    """Create a mock text part."""
    part = MagicMock()
    part.text = text
    part.function_call = None
    return part


def make_function_call_part(name: str = "get_weather", args: dict | None = None):
    """Create a mock function_call part."""
    part = MagicMock()
    part.text = None
    part.function_call = MagicMock()
    part.function_call.name = name
    part.function_call.args = args or {"location": "London"}
    return part


def make_response(*parts):
    """Create a mock Gemini response."""
    content = MagicMock()
    content.role = "model"
    content.parts = list(parts)

    candidate = MagicMock()
    candidate.content = content

    response = MagicMock()
    response.candidates = [candidate]
    return response


class TestGoogleThreadAgent:
    """Tests for instantiation and metadata."""

    def test_instantiation(self):
        mock_client = MagicMock()
        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()])
        assert isinstance(agent, GoogleThreadAgent)

    def test_custom_name(self):
        mock_client = MagicMock()
        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()], name="my-thread-agent")
        assert agent.name == "my-thread-agent"

    def test_custom_model(self):
        mock_client = MagicMock()
        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()], model="gemini-2.5-pro")
        assert agent.model == "gemini-2.5-pro"

    def test_default_values(self):
        mock_client = MagicMock()
        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()])
        assert agent.name == "google-thread-agent"
        assert agent.model == "gemini-2.5-flash"

    def test_thread_type_metadata(self):
        mock_client = MagicMock()
        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()])
        assert agent.metadata["type"] == "thread"
        assert agent.metadata["input"] == AGENT_TYPES["thread"]["input"]
        assert agent.metadata["output"] == AGENT_TYPES["thread"]["output"]
        assert agent.metadata["capabilities"]["streaming"] is False


class TestGoogleThreadAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_no_tool_calls(self):
        """invoke() should return messages when no tool calls."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=make_response(make_text_part("Hello!"))
        )

        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        result = await agent.invoke(request)

        assert "output" in result
        messages = result["output"]
        assert len(messages) >= 2
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_invoke_with_tool_calls(self):
        """invoke() should execute tool calls and return full thread."""
        mock_client = MagicMock()

        # First call returns function_call, second returns final text
        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[
                make_response(
                    make_text_part("Let me check the weather."),
                    make_function_call_part("get_weather", {"location": "London"}),
                ),
                make_response(make_text_part("The weather in London is sunny, 22°C.")),
            ]
        )

        tool = make_mock_tool()
        agent = GoogleThreadAgent(mock_client, tools=[tool])
        request = AgentRequest(
            input={"messages": [{"role": "user", "content": "What's the weather in London?"}]}
        )

        result = await agent.invoke(request)

        messages = result["output"]
        # user -> assistant(tool_call) -> tool(result) -> assistant(final)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["tool_calls"] is not None
        assert messages[2]["role"] == "tool"
        assert messages[3]["role"] == "assistant"

        # Verify tool was called
        tool.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_tool_error_handling(self):
        """invoke() should handle tool errors gracefully."""
        mock_client = MagicMock()

        mock_client.aio.models.generate_content = AsyncMock(
            side_effect=[
                make_response(
                    make_function_call_part("get_weather", {"location": "London"}),
                ),
                make_response(make_text_part("Sorry, I couldn't get the weather.")),
            ]
        )

        tool = make_mock_tool()
        tool.call = AsyncMock(side_effect=Exception("API timeout"))
        agent = GoogleThreadAgent(mock_client, tools=[tool])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Weather?"}]})

        result = await agent.invoke(request)

        messages = result["output"]
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        assert "error" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_invoke_respects_max_turns(self):
        """invoke() should stop after max_turns iterations."""
        mock_client = MagicMock()

        # Always return function_call to test max_turns limit
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=make_response(
                make_function_call_part("get_weather", {"location": "London"}),
            )
        )

        tool = make_mock_tool()
        agent = GoogleThreadAgent(mock_client, tools=[tool], max_turns=3)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Loop"}]})

        await agent.invoke(request)

        assert mock_client.aio.models.generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_invoke_passes_model(self):
        """invoke() should use the configured model."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=make_response(make_text_part("Hi"))
        )

        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()], model="gemini-2.5-pro")
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_invoke_passes_tool_definitions(self):
        """invoke() should pass tool definitions to the client."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=make_response(make_text_part("Hi"))
        )

        tool = make_mock_tool("get_weather")
        agent = GoogleThreadAgent(mock_client, tools=[tool])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        tools = call_kwargs["config"]["tools"]
        assert len(tools[0]["function_declarations"]) == 1
        assert tools[0]["function_declarations"][0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_invoke_extracts_system_message(self):
        """invoke() should extract system message for Gemini API."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=make_response(make_text_part("Hi"))
        )

        agent = GoogleThreadAgent(mock_client, tools=[make_mock_tool()])
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are a weather assistant."},
                    {"role": "user", "content": "Hi"},
                ]
            }
        )

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"]["system_instruction"] == "You are a weather assistant."
