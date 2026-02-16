"""Tests for the LangChain chat agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from reminix_langchain import LangChainChatAgent
from reminix_runtime import AGENT_TYPES, AgentRequest


class TestLangChainChatAgent:
    """Tests for the LangChainChatAgent class."""

    def test_instantiation(self):
        """LangChainChatAgent should be instantiable."""
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainChatAgent(mock_runnable)

        assert isinstance(agent, LangChainChatAgent)

    def test_custom_name(self):
        """LangChainChatAgent should accept a custom name."""
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainChatAgent(mock_runnable, name="my-custom-agent")

        assert agent.name == "my-custom-agent"

    def test_default_name(self):
        """LangChainChatAgent should use default name if not provided."""
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainChatAgent(mock_runnable)

        assert agent.name == "langchain-agent"

    def test_chat_type_metadata(self):
        """LangChainChatAgent should have chat type metadata."""
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainChatAgent(mock_runnable)

        assert agent.metadata["type"] == "chat"
        assert agent.metadata["input"] == AGENT_TYPES["chat"]["input"]


class TestLangChainChatAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_runnable(self):
        """invoke() should call the underlying runnable with the input."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello!"))

        agent = LangChainChatAgent(mock_runnable)
        request = AgentRequest(input={"query": "What is AI?"})

        await agent.invoke(request)

        mock_runnable.ainvoke.assert_called_once_with({"query": "What is AI?"})

    @pytest.mark.asyncio
    async def test_invoke_returns_output(self):
        """invoke() should return the output from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Hello from LangChain!"))

        agent = LangChainChatAgent(mock_runnable)
        request = AgentRequest(input={"query": "Hi"})

        response = await agent.invoke(request)

        assert response["output"] == "Hello from LangChain!"

    @pytest.mark.asyncio
    async def test_invoke_handles_dict_response(self):
        """invoke() should handle dict responses from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value={"result": "success", "value": 42})

        agent = LangChainChatAgent(mock_runnable)
        request = AgentRequest(input={"task": "compute"})

        response = await agent.invoke(request)

        assert response["output"] == {"result": "success", "value": 42}

    @pytest.mark.asyncio
    async def test_invoke_handles_string_response(self):
        """invoke() should handle string responses from the runnable."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value="Simple string result")

        agent = LangChainChatAgent(mock_runnable)
        request = AgentRequest(input={"query": "test"})

        response = await agent.invoke(request)

        assert response["output"] == "Simple string result"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should convert messages to LangChain format."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        agent = LangChainChatAgent(mock_runnable)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                    {"role": "user", "content": "How are you?"},
                ]
            }
        )

        await agent.invoke(request)

        call_args = mock_runnable.ainvoke.call_args[0][0]
        assert len(call_args) == 4
        assert isinstance(call_args[0], SystemMessage)
        assert isinstance(call_args[1], HumanMessage)
        assert isinstance(call_args[2], AIMessage)
        assert isinstance(call_args[3], HumanMessage)


class TestMessageConversion:
    """Tests for message format conversion."""

    @pytest.mark.asyncio
    async def test_tool_role_handled(self):
        """Tool role messages should be handled appropriately."""
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        agent = LangChainChatAgent(mock_runnable)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "user", "content": "Use a tool"},
                    {"role": "tool", "content": "Tool result"},
                ]
            }
        )

        response = await agent.invoke(request)
        assert response["output"] == "Response"
