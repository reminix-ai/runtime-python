"""Tests for the LangGraph thread adapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from reminix_langgraph import LangGraphThreadAgent
from reminix_runtime import AGENT_TEMPLATES, AgentRequest


class TestLangGraphThreadAgent:
    """Tests for the LangGraphThreadAgent class."""

    def test_instantiation(self):
        """LangGraphThreadAgent should be instantiable."""
        mock_graph = MagicMock()
        agent = LangGraphThreadAgent(mock_graph)

        assert isinstance(agent, LangGraphThreadAgent)

    def test_custom_name(self):
        """LangGraphThreadAgent should accept a custom name."""
        mock_graph = MagicMock()
        agent = LangGraphThreadAgent(mock_graph, name="my-custom-agent")

        assert agent.name == "my-custom-agent"

    def test_default_name(self):
        """LangGraphThreadAgent should use default name if not provided."""
        mock_graph = MagicMock()
        agent = LangGraphThreadAgent(mock_graph)

        assert agent.name == "langgraph-agent"

    def test_thread_template_metadata(self):
        """LangGraphThreadAgent should have thread template metadata."""
        mock_graph = MagicMock()
        agent = LangGraphThreadAgent(mock_graph)

        assert agent.metadata["template"] == "thread"
        assert agent.metadata["input"] == AGENT_TEMPLATES["thread"]["input"]


class TestLangGraphThreadAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_invoke_calls_graph(self):
        """invoke() should call the underlying graph with the input."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hello!")]})

        agent = LangGraphThreadAgent(mock_graph)
        request = AgentRequest(input={"query": "What is AI?"})

        await agent.invoke(request)

        mock_graph.ainvoke.assert_called_once_with({"query": "What is AI?"})

    @pytest.mark.asyncio
    async def test_invoke_returns_output_from_messages(self):
        """invoke() should extract output from messages in the result."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [HumanMessage(content="Hello"), AIMessage(content="Hi there!")]
            }
        )

        agent = LangGraphThreadAgent(mock_graph)
        request = AgentRequest(input={"messages": []})

        response = await agent.invoke(request)

        assert response["output"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_invoke_handles_dict_result(self):
        """invoke() should handle dict results without messages."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"result": "success"})

        agent = LangGraphThreadAgent(mock_graph)
        request = AgentRequest(input={"task": "compute"})

        response = await agent.invoke(request)

        assert response["output"] == {"result": "success"}

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """invoke() should call graph with state dict format for chat-style input."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"messages": [AIMessage(content="Hello!")]})

        agent = LangGraphThreadAgent(mock_graph)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})

        await agent.invoke(request)

        call_args = mock_graph.ainvoke.call_args[0][0]
        assert "messages" in call_args
        assert len(call_args["messages"]) == 1
        assert isinstance(call_args["messages"][0], HumanMessage)

    @pytest.mark.asyncio
    async def test_invoke_converts_messages_correctly(self):
        """invoke() should convert messages to/from LangChain format."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    SystemMessage(content="You are helpful"),
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi!"),
                ]
            }
        )

        agent = LangGraphThreadAgent(mock_graph)
        request = AgentRequest(
            input={
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ]
            }
        )

        response = await agent.invoke(request)

        assert response["output"] == "Hi!"
