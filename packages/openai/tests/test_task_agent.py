"""Tests for the OpenAI task agent."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_openai import OpenAITaskAgent
from reminix_runtime import AGENT_TEMPLATES, AgentRequest

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}


class TestOpenAITaskAgent:
    """Tests for the OpenAITaskAgent class."""

    def test_instantiation(self):
        """OpenAITaskAgent should be instantiable."""
        mock_client = MagicMock()
        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA)

        assert isinstance(agent, OpenAITaskAgent)

    def test_custom_name(self):
        """OpenAITaskAgent should accept a custom name."""
        mock_client = MagicMock()
        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA, name="my-task-agent")

        assert agent.name == "my-task-agent"

    def test_custom_model(self):
        """OpenAITaskAgent should accept a custom model."""
        mock_client = MagicMock()
        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA, model="gpt-4o")

        assert agent.model == "gpt-4o"

    def test_default_values(self):
        """OpenAITaskAgent should use default values if not provided."""
        mock_client = MagicMock()
        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA)

        assert agent.name == "openai-task-agent"
        assert agent.model == "gpt-4o-mini"

    def test_task_template_metadata(self):
        """OpenAITaskAgent should have task template metadata."""
        mock_client = MagicMock()
        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA)

        assert agent.metadata["template"] == "task"
        assert agent.metadata["input"] == AGENT_TEMPLATES["task"]["input"]
        assert agent.metadata["output"] == AGENT_TEMPLATES["task"]["output"]
        assert agent.metadata["capabilities"]["streaming"] is False


class TestOpenAITaskAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content=json.dumps({"sentiment": "positive", "confidence": 0.95}))
            )
        ]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze sentiment of: I love this!"})

        await agent.invoke(request)

        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_structured_output(self):
        """invoke() should return parsed JSON output."""
        mock_client = MagicMock()
        result = {"sentiment": "positive", "confidence": 0.95}
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(result)))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze sentiment"})

        response = await agent.invoke(request)

        assert response["output"] == result

    @pytest.mark.asyncio
    async def test_invoke_uses_response_format(self):
        """invoke() should use json_schema response format."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="{}"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Do something"})

        await agent.invoke(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["response_format"]["type"] == "json_schema"
        assert call_kwargs["response_format"]["json_schema"]["schema"] == SAMPLE_SCHEMA

    @pytest.mark.asyncio
    async def test_invoke_passes_model(self):
        """invoke() should use the configured model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="{}"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA, model="gpt-4o")
        request = AgentRequest(input={"task": "Do something"})

        await agent.invoke(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_invoke_includes_extra_context(self):
        """invoke() should include additional input fields as context."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="{}"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAITaskAgent(mock_client, SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze", "text": "Hello world", "language": "en"})

        await agent.invoke(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        prompt = call_kwargs["messages"][0]["content"]
        assert "Hello world" in prompt
        assert "language" in prompt
