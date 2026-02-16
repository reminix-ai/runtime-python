"""Tests for the Google Gemini chat agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reminix_google import GoogleChatAgent
from reminix_runtime import AGENT_TYPES, AgentRequest


class TestGoogleChatAgent:
    """Tests for the GoogleChatAgent class."""

    def test_instantiation(self):
        """GoogleChatAgent should be instantiable."""
        mock_client = MagicMock()
        agent = GoogleChatAgent(mock_client)

        assert isinstance(agent, GoogleChatAgent)

    def test_custom_name(self):
        """GoogleChatAgent should accept a custom name."""
        mock_client = MagicMock()
        agent = GoogleChatAgent(mock_client, name="my-custom-agent")

        assert agent.name == "my-custom-agent"

    def test_custom_model(self):
        """GoogleChatAgent should accept a custom model."""
        mock_client = MagicMock()
        agent = GoogleChatAgent(mock_client, model="gemini-2.5-pro")

        assert agent.model == "gemini-2.5-pro"

    def test_default_values(self):
        """GoogleChatAgent should use default values if not provided."""
        mock_client = MagicMock()
        agent = GoogleChatAgent(mock_client)

        assert agent.name == "google-agent"
        assert agent.model == "gemini-2.5-flash"

    def test_chat_type_metadata(self):
        """GoogleChatAgent should have chat type metadata."""
        mock_client = MagicMock()
        agent = GoogleChatAgent(mock_client)

        assert agent.metadata["type"] == "chat"
        assert agent.metadata["input"] == AGENT_TYPES["chat"]["input"]


class TestGoogleChatAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_client(self):
        """invoke() should call the Gemini client."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello!"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = GoogleChatAgent(mock_client)
        request = AgentRequest(input={"prompt": "Hi"})

        await agent.invoke(request)

        mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Hello from Gemini!"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = GoogleChatAgent(mock_client)
        request = AgentRequest(input={"prompt": "Hi"})

        response = await agent.invoke(request)

        assert response["output"] == "Hello from Gemini!"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should handle input with messages key."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = GoogleChatAgent(mock_client)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        response = await agent.invoke(request)

        assert response["output"] == "Response"

    @pytest.mark.asyncio
    async def test_invoke_extracts_system_message(self):
        """invoke() should extract system message as system_instruction."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = GoogleChatAgent(mock_client)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hi"},
                ]
            }
        )

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["config"]["system_instruction"] == "You are helpful"
        assert all(c["role"] != "system" for c in call_kwargs["contents"])

    @pytest.mark.asyncio
    async def test_invoke_maps_assistant_to_model(self):
        """invoke() should map assistant role to model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

        agent = GoogleChatAgent(mock_client)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "How are you?"},
                ]
            }
        )

        await agent.invoke(request)

        call_kwargs = mock_client.aio.models.generate_content.call_args[1]
        assert call_kwargs["contents"][1]["role"] == "model"
