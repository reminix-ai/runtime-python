"""Tests for the Anthropic task agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_anthropic import AnthropicTaskAgent
from reminix_runtime import AGENT_TEMPLATES, AgentRequest

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}


class TestAnthropicTaskAgent:
    """Tests for the AnthropicTaskAgent class."""

    def test_instantiation(self):
        """AnthropicTaskAgent should be instantiable."""
        mock_client = MagicMock()
        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)

        assert isinstance(agent, AnthropicTaskAgent)

    def test_custom_name(self):
        """AnthropicTaskAgent should accept a custom name."""
        mock_client = MagicMock()
        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA, name="my-task-agent")

        assert agent.name == "my-task-agent"

    def test_custom_model(self):
        """AnthropicTaskAgent should accept a custom model."""
        mock_client = MagicMock()
        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA, model="claude-opus-4-20250514")

        assert agent.model == "claude-opus-4-20250514"

    def test_default_values(self):
        """AnthropicTaskAgent should use default values if not provided."""
        mock_client = MagicMock()
        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)

        assert agent.name == "anthropic-task-agent"
        assert agent.model == "claude-sonnet-4-20250514"

    def test_task_template_metadata(self):
        """AnthropicTaskAgent should have task template metadata."""
        mock_client = MagicMock()
        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)

        assert agent.metadata["template"] == "task"
        assert agent.metadata["input"] == AGENT_TEMPLATES["task"]["input"]
        assert agent.metadata["output"] == AGENT_TEMPLATES["task"]["output"]
        assert agent.metadata["capabilities"]["streaming"] is False


class TestAnthropicTaskAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [
            MagicMock(type="tool_use", input={"sentiment": "positive", "confidence": 0.95})
        ]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze sentiment of: I love this!"})

        await agent.invoke(request)

        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_structured_output(self):
        """invoke() should return structured output from tool_use block."""
        mock_client = MagicMock()
        result = {"sentiment": "positive", "confidence": 0.95}
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="tool_use", input=result)]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze sentiment"})

        response = await agent.invoke(request)

        assert response["output"] == result

    @pytest.mark.asyncio
    async def test_invoke_uses_tool_choice(self):
        """invoke() should force tool_use with tool_choice."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="tool_use", input={})]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Do something"})

        await agent.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tool_choice"] == {"type": "tool", "name": "task_result"}
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "task_result"
        assert call_kwargs["tools"][0]["input_schema"] == SAMPLE_SCHEMA

    @pytest.mark.asyncio
    async def test_invoke_passes_model(self):
        """invoke() should use the configured model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="tool_use", input={})]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA, model="claude-opus-4-20250514")
        request = AgentRequest(input={"task": "Do something"})

        await agent.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-20250514"

    @pytest.mark.asyncio
    async def test_invoke_includes_extra_context(self):
        """invoke() should include additional input fields as context."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="tool_use", input={})]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze", "text": "Hello world", "language": "en"})

        await agent.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "Hello world" in prompt
        assert "language" in prompt

    @pytest.mark.asyncio
    async def test_invoke_returns_empty_on_no_tool_use(self):
        """invoke() should return empty dict if no tool_use block found."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Some text")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicTaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Do something"})

        response = await agent.invoke(request)

        assert response["output"] == {}
