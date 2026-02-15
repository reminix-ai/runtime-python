"""Tests for the Anthropic chat adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_anthropic import AnthropicChatAgent
from reminix_runtime import AGENT_TEMPLATES, AgentRequest


class TestAnthropicChatAgent:
    """Tests for the AnthropicChatAgent class."""

    def test_instantiation(self):
        """AnthropicChatAgent should be instantiable."""
        mock_client = MagicMock()
        agent = AnthropicChatAgent(mock_client)

        assert isinstance(agent, AnthropicChatAgent)

    def test_custom_name(self):
        """AnthropicChatAgent should accept a custom name."""
        mock_client = MagicMock()
        agent = AnthropicChatAgent(mock_client, name="my-custom-agent")

        assert agent.name == "my-custom-agent"

    def test_custom_model(self):
        """AnthropicChatAgent should accept a custom model."""
        mock_client = MagicMock()
        agent = AnthropicChatAgent(mock_client, model="claude-opus-4-20250514")

        assert agent.model == "claude-opus-4-20250514"

    def test_default_values(self):
        """AnthropicChatAgent should use default values if not provided."""
        mock_client = MagicMock()
        agent = AnthropicChatAgent(mock_client)

        assert agent.name == "anthropic-agent"
        assert agent.model == "claude-sonnet-4-20250514"

    def test_chat_template_metadata(self):
        """AnthropicChatAgent should have chat template metadata."""
        mock_client = MagicMock()
        agent = AnthropicChatAgent(mock_client)

        assert agent.metadata["template"] == "chat"
        assert agent.metadata["input"] == AGENT_TEMPLATES["chat"]["input"]


class TestAnthropicChatAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the Anthropic client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicChatAgent(mock_client)
        request = AgentRequest(input={"prompt": "Hi"})

        await agent.invoke(request)

        mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Hello from Anthropic!")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicChatAgent(mock_client)
        request = AgentRequest(input={"prompt": "Hi"})

        response = await agent.invoke(request)

        assert response["output"] == "Hello from Anthropic!"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should handle input with messages key."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicChatAgent(mock_client)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        response = await agent.invoke(request)

        assert response["output"] == "Response"

    @pytest.mark.asyncio
    async def test_invoke_extracts_system_message(self):
        """invoke() should extract system message for Anthropic API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock(type="text", text="Response")]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        agent = AnthropicChatAgent(mock_client)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ]
            }
        )

        await agent.invoke(request)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "You are helpful"
        assert all(m["role"] != "system" for m in call_kwargs["messages"])
