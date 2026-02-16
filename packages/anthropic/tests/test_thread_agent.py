"""Tests for the Anthropic thread agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_anthropic import AnthropicThreadAgent
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


def make_text_block(text: str = "Hello!"):
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def make_tool_use_block(
    block_id: str = "toolu_1", name: str = "get_weather", input_data: dict | None = None
):
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    block.input = input_data or {"location": "London"}
    return block


def make_response(*blocks):
    """Create a mock Anthropic response."""
    response = MagicMock()
    response.content = list(blocks)
    return response


class TestAnthropicThreadAgent:
    """Tests for instantiation and metadata."""

    def test_instantiation(self):
        mock_client = MagicMock()
        agent = AnthropicThreadAgent(mock_client, tools=[make_mock_tool()])
        assert isinstance(agent, AnthropicThreadAgent)

    def test_custom_name(self):
        mock_client = MagicMock()
        agent = AnthropicThreadAgent(mock_client, tools=[make_mock_tool()], name="my-thread-agent")
        assert agent.name == "my-thread-agent"

    def test_custom_model(self):
        mock_client = MagicMock()
        agent = AnthropicThreadAgent(
            mock_client, tools=[make_mock_tool()], model="claude-opus-4-20250514"
        )
        assert agent.model == "claude-opus-4-20250514"

    def test_default_values(self):
        mock_client = MagicMock()
        agent = AnthropicThreadAgent(mock_client, tools=[make_mock_tool()])
        assert agent.name == "anthropic-thread-agent"
        assert agent.model == "claude-sonnet-4-5-20250929"

    def test_thread_type_metadata(self):
        mock_client = MagicMock()
        agent = AnthropicThreadAgent(mock_client, tools=[make_mock_tool()])
        assert agent.metadata["type"] == "thread"
        assert agent.metadata["input"] == AGENT_TYPES["thread"]["input"]
        assert agent.metadata["output"] == AGENT_TYPES["thread"]["output"]
        assert agent.metadata["capabilities"]["streaming"] is False


class TestAnthropicThreadAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_no_tool_calls(self):
        """invoke() should return messages when no tool calls."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(
            return_value=make_response(make_text_block("Hello!"))
        )

        agent = AnthropicThreadAgent(mock_client, tools=[make_mock_tool()])
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

        # First call returns tool_use, second returns final text
        mock_client.messages.create = AsyncMock(
            side_effect=[
                make_response(
                    make_text_block("Let me check the weather."),
                    make_tool_use_block("toolu_1", "get_weather", {"location": "London"}),
                ),
                make_response(make_text_block("The weather in London is sunny, 22°C.")),
            ]
        )

        tool = make_mock_tool()
        agent = AnthropicThreadAgent(mock_client, tools=[tool])
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

        mock_client.messages.create = AsyncMock(
            side_effect=[
                make_response(
                    make_tool_use_block("toolu_1", "get_weather", {"location": "London"}),
                ),
                make_response(make_text_block("Sorry, I couldn't get the weather.")),
            ]
        )

        tool = make_mock_tool()
        tool.call = AsyncMock(side_effect=Exception("API timeout"))
        agent = AnthropicThreadAgent(mock_client, tools=[tool])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Weather?"}]})

        result = await agent.invoke(request)

        messages = result["output"]
        tool_msg = [m for m in messages if m["role"] == "tool"][0]
        assert "Error" in tool_msg["content"]

    @pytest.mark.asyncio
    async def test_invoke_respects_max_turns(self):
        """invoke() should stop after max_turns iterations."""
        mock_client = MagicMock()

        # Always return tool_use to test max_turns limit
        mock_client.messages.create = AsyncMock(
            return_value=make_response(
                make_tool_use_block("toolu_1", "get_weather", {"location": "London"}),
            )
        )

        tool = make_mock_tool()
        agent = AnthropicThreadAgent(mock_client, tools=[tool], max_turns=3)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Loop"}]})

        await agent.invoke(request)

        assert mock_client.messages.create.call_count == 3

    @pytest.mark.asyncio
    async def test_invoke_passes_model(self):
        """invoke() should use the configured model."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=make_response(make_text_block("Hi")))

        agent = AnthropicThreadAgent(
            mock_client, tools=[make_mock_tool()], model="claude-opus-4-20250514"
        )
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_invoke_passes_tool_definitions(self):
        """invoke() should pass tool definitions to the client."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=make_response(make_text_block("Hi")))

        tool = make_mock_tool("get_weather")
        agent = AnthropicThreadAgent(mock_client, tools=[tool])
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        tools = call_kwargs["tools"]
        assert len(tools) == 1
        assert tools[0]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_invoke_extracts_system_message(self):
        """invoke() should extract system message for Anthropic API."""
        mock_client = MagicMock()
        mock_client.messages.create = AsyncMock(return_value=make_response(make_text_block("Hi")))

        agent = AnthropicThreadAgent(mock_client, tools=[make_mock_tool()])
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are a weather assistant."},
                    {"role": "user", "content": "Hi"},
                ]
            }
        )

        await agent.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are a weather assistant."
