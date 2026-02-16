"""Tests for the Google Gemini task agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_google import GoogleTaskAgent
from reminix_runtime import AGENT_TYPES, AgentRequest

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}


def _make_fc_response(args=None):
    """Create a mock response with a function call."""
    mock_fc = MagicMock()
    mock_fc.name = "task_result"
    mock_fc.args = args or {}

    mock_part = MagicMock()
    mock_part.function_call = mock_fc
    mock_part.text = None

    mock_content = MagicMock()
    mock_content.parts = [mock_part]

    mock_candidate = MagicMock()
    mock_candidate.content = mock_content

    mock_response = MagicMock()
    mock_response.candidates = [mock_candidate]
    return mock_response


class TestGoogleTaskAgent:
    """Tests for the GoogleTaskAgent class."""

    def test_instantiation(self):
        """GoogleTaskAgent should be instantiable."""
        mock_client = MagicMock()
        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)

        assert isinstance(agent, GoogleTaskAgent)

    def test_custom_name(self):
        """GoogleTaskAgent should accept a custom name."""
        mock_client = MagicMock()
        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA, name="my-task-agent")

        assert agent.name == "my-task-agent"

    def test_custom_model(self):
        """GoogleTaskAgent should accept a custom model."""
        mock_client = MagicMock()
        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA, model="gemini-2.5-pro")

        assert agent.model == "gemini-2.5-pro"

    def test_default_values(self):
        """GoogleTaskAgent should use default values if not provided."""
        mock_client = MagicMock()
        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)

        assert agent.name == "google-task-agent"
        assert agent.model == "gemini-2.5-flash"

    def test_task_type_metadata(self):
        """GoogleTaskAgent should have task type metadata."""
        mock_client = MagicMock()
        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)

        assert agent.metadata["type"] == "task"
        assert agent.metadata["input"] == AGENT_TYPES["task"]["input"]
        assert agent.metadata["output"] == AGENT_TYPES["task"]["output"]
        assert agent.metadata["capabilities"]["streaming"] is False


class TestGoogleTaskAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the Gemini client."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(
            return_value=_make_fc_response({"sentiment": "positive", "confidence": 0.95})
        )

        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze sentiment of: I love this!"})

        await agent.invoke(request)

        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_structured_output(self):
        """invoke() should return structured output from function call."""
        result = {"sentiment": "positive", "confidence": 0.95}
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_make_fc_response(result))

        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze sentiment"})

        response = await agent.invoke(request)

        assert response["output"] == result

    @pytest.mark.asyncio
    async def test_invoke_uses_function_calling_config(self):
        """invoke() should force function calling with tool_config."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_make_fc_response({}))

        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Do something"})

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        config = call_kwargs["config"]
        assert config["tool_config"]["function_calling_config"]["mode"] == "ANY"
        assert config["tool_config"]["function_calling_config"]["allowed_function_names"] == [
            "task_result"
        ]
        assert len(config["tools"][0]["function_declarations"]) == 1
        assert config["tools"][0]["function_declarations"][0]["name"] == "task_result"
        assert config["tools"][0]["function_declarations"][0]["parameters"] == SAMPLE_SCHEMA

    @pytest.mark.asyncio
    async def test_invoke_passes_model(self):
        """invoke() should use the configured model."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_make_fc_response({}))

        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA, model="gemini-2.5-pro")
        request = AgentRequest(input={"task": "Do something"})

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["model"] == "gemini-2.5-pro"

    @pytest.mark.asyncio
    async def test_invoke_includes_extra_context(self):
        """invoke() should include additional input fields as context."""
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock(return_value=_make_fc_response({}))

        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Analyze", "text": "Hello world", "language": "en"})

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        prompt = call_kwargs["contents"][0]["parts"][0]["text"]
        assert "Hello world" in prompt
        assert "language" in prompt

    @pytest.mark.asyncio
    async def test_invoke_returns_empty_on_no_function_call(self):
        """invoke() should return empty dict if no function call found."""
        mock_client = MagicMock()

        mock_part = MagicMock()
        mock_part.function_call = None
        mock_part.text = "Some text"

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = GoogleTaskAgent(mock_client, output_schema=SAMPLE_SCHEMA)
        request = AgentRequest(input={"task": "Do something"})

        response = await agent.invoke(request)

        assert response["output"] == {}
