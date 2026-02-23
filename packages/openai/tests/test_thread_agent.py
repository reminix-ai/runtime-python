"""Tests for the OpenAI thread agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_openai import OpenAIThreadAgent
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


def make_response(content: str = "Hello!", tool_calls: list | None = None):
    """Create a mock OpenAI response."""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


def make_tool_call(call_id: str = "call_1", name: str = "get_weather", arguments: str = "{}"):
    """Create a mock tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class TestOpenAIThreadAgent:
    """Tests for instantiation and metadata."""

    def test_instantiation(self):
        mock_client = MagicMock()
        agent = OpenAIThreadAgent(mock_client, tools=[make_mock_tool()])
        assert isinstance(agent, OpenAIThreadAgent)

    def test_custom_name(self):
        mock_client = MagicMock()
        agent = OpenAIThreadAgent(mock_client, tools=[make_mock_tool()], name="my-thread-agent")
        assert agent.name == "my-thread-agent"

    def test_custom_model(self):
        mock_client = MagicMock()
        agent = OpenAIThreadAgent(mock_client, tools=[make_mock_tool()], model="gpt-4o")
        assert agent.model == "gpt-4o"

    def test_default_values(self):
        mock_client = MagicMock()
        agent = OpenAIThreadAgent(mock_client, tools=[make_mock_tool()])
        assert agent.name == "openai-thread-agent"
        assert agent.model == "gpt-4o-mini"

    def test_thread_type_metadata(self):
        mock_client = MagicMock()
        agent = OpenAIThreadAgent(mock_client, tools=[make_mock_tool()])
        assert agent.metadata["type"] == "thread"
        assert agent.metadata["inputSchema"] == AGENT_TYPES["thread"]["inputSchema"]
        assert agent.metadata["outputSchema"] == AGENT_TYPES["thread"]["outputSchema"]
        assert agent.metadata["capabilities"]["streaming"] is False


class TestOpenAIThreadAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_no_tool_calls(self):
        """invoke() should return messages when no tool calls."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=make_response("Hello!"))

        agent = OpenAIThreadAgent(mock_client, tools=[make_mock_tool()])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        result = await agent.invoke(request)

        assert "output" in result
        messages = result["output"]
        # Should have input message + assistant response
        assert len(messages) >= 2
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_invoke_with_tool_calls(self):
        """invoke() should execute tool calls and return full thread."""
        mock_client = MagicMock()
        tc = make_tool_call(
            call_id="call_1",
            name="get_weather",
            arguments=json.dumps({"location": "London"}),
        )

        # First call returns tool call, second returns final response
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                make_response("", tool_calls=[tc]),
                make_response("The weather in London is sunny, 22°C."),
            ]
        )

        tool = make_mock_tool()
        agent = OpenAIThreadAgent(mock_client, tools=[tool])
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
        assert "sunny" in messages[3]["content"].lower() or "22" in messages[3]["content"]

        # Verify tool was called
        tool.call.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_tool_error_handling(self):
        """invoke() should handle tool errors gracefully."""
        mock_client = MagicMock()
        tc = make_tool_call(
            call_id="call_1",
            name="get_weather",
            arguments=json.dumps({"location": "London"}),
        )

        mock_client.chat.completions.create = AsyncMock(
            side_effect=[
                make_response("", tool_calls=[tc]),
                make_response("Sorry, I couldn't get the weather."),
            ]
        )

        tool = make_mock_tool()
        tool.call = AsyncMock(side_effect=Exception("API timeout"))
        agent = OpenAIThreadAgent(mock_client, tools=[tool])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Weather?"}]})

        result = await agent.invoke(request)

        messages = result["output"]
        # Tool result should contain error
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        assert "Error" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_invoke_respects_max_turns(self):
        """invoke() should stop after max_turns iterations."""
        mock_client = MagicMock()
        tc = make_tool_call(
            call_id="call_1",
            name="get_weather",
            arguments="{}",
        )

        # Always return tool calls to test max_turns limit
        mock_client.chat.completions.create = AsyncMock(
            return_value=make_response("", tool_calls=[tc])
        )

        tool = make_mock_tool()
        agent = OpenAIThreadAgent(mock_client, tools=[tool], max_turns=3)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Loop"}]})

        await agent.invoke(request)

        # Should have called create exactly 3 times (max_turns=3)
        assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_invoke_passes_model(self):
        """invoke() should use the configured model."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=make_response("Hi"))

        agent = OpenAIThreadAgent(mock_client, tools=[make_mock_tool()], model="gpt-4o")
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_invoke_passes_tool_definitions(self):
        """invoke() should pass tool definitions to the client."""
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=make_response("Hi"))

        tool = make_mock_tool("get_weather")
        agent = OpenAIThreadAgent(mock_client, tools=[tool])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"
