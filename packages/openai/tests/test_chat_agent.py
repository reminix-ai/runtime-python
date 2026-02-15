"""Tests for the OpenAI chat agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_openai import OpenAIChatAgent
from reminix_runtime import AGENT_TEMPLATES, AgentRequest


class TestOpenAIChatAgent:
    """Tests for the OpenAIChatAgent class."""

    def test_instantiation(self):
        """OpenAIChatAgent should be instantiable."""
        mock_client = MagicMock()
        agent = OpenAIChatAgent(mock_client)

        assert isinstance(agent, OpenAIChatAgent)

    def test_custom_name(self):
        """OpenAIChatAgent should accept a custom name."""
        mock_client = MagicMock()
        agent = OpenAIChatAgent(mock_client, name="my-custom-agent")

        assert agent.name == "my-custom-agent"

    def test_custom_model(self):
        """OpenAIChatAgent should accept a custom model."""
        mock_client = MagicMock()
        agent = OpenAIChatAgent(mock_client, model="gpt-4o")

        assert agent.model == "gpt-4o"

    def test_default_values(self):
        """OpenAIChatAgent should use default values if not provided."""
        mock_client = MagicMock()
        agent = OpenAIChatAgent(mock_client)

        assert agent.name == "openai-agent"
        assert agent.model == "gpt-4o-mini"

    def test_chat_template_metadata(self):
        """OpenAIChatAgent should have chat template metadata."""
        mock_client = MagicMock()
        agent = OpenAIChatAgent(mock_client)

        assert agent.metadata["template"] == "chat"
        assert agent.metadata["input"] == AGENT_TEMPLATES["chat"]["input"]


class TestOpenAIChatAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the OpenAI client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAIChatAgent(mock_client)
        request = AgentRequest(input={"prompt": "Hi"})

        await agent.invoke(request)

        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI!"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAIChatAgent(mock_client)
        request = AgentRequest(input={"prompt": "Hi"})

        response = await agent.invoke(request)

        assert response["output"] == "Hello from OpenAI!"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should handle input with messages key."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAIChatAgent(mock_client)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_invoke_passes_model(self):
        """invoke() should use the configured model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        agent = OpenAIChatAgent(mock_client, model="gpt-4o")
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
