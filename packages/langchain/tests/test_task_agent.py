"""Tests for the LangChain task agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from reminix_langchain import LangChainTaskAgent
from reminix_runtime import AGENT_TYPES, AgentRequest


class TestLangChainTaskAgent:
    """Tests for the LangChainTaskAgent class."""

    def test_instantiation_with_runnable(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainTaskAgent(mock_runnable)
        assert isinstance(agent, LangChainTaskAgent)

    def test_instantiation_with_graph(self):
        mock_graph = MagicMock(spec=Runnable)
        mock_graph.get_graph = MagicMock()
        agent = LangChainTaskAgent(mock_graph)
        assert isinstance(agent, LangChainTaskAgent)

    def test_custom_name(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainTaskAgent(mock_runnable, name="my-task")
        assert agent.name == "my-task"

    def test_default_name(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainTaskAgent(mock_runnable)
        assert agent.name == "langchain-task-agent"

    def test_task_type_metadata(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainTaskAgent(mock_runnable)
        assert agent.metadata["type"] == "task"
        assert agent.metadata["inputSchema"] == AGENT_TYPES["task"]["inputSchema"]
        assert agent.metadata["outputSchema"] == AGENT_TYPES["task"]["outputSchema"]


class TestLangChainTaskAgentGraph:
    """Tests for invoke() with CompiledStateGraph."""

    @pytest.mark.asyncio
    async def test_invoke_graph_parses_json(self):
        mock_graph = MagicMock(spec=Runnable)
        mock_graph.get_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Summarize this"),
                    AIMessage(content='{"summary": "done"}'),
                ]
            }
        )

        agent = LangChainTaskAgent(mock_graph)
        request = AgentRequest(input={"task": "Summarize this"})

        result = await agent.invoke(request)

        assert result["output"] == {"summary": "done"}

    @pytest.mark.asyncio
    async def test_invoke_graph_returns_plain_text(self):
        mock_graph = MagicMock(spec=Runnable)
        mock_graph.get_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Do it"),
                    AIMessage(content="Just plain text"),
                ]
            }
        )

        agent = LangChainTaskAgent(mock_graph)
        request = AgentRequest(input={"task": "Do it"})

        result = await agent.invoke(request)

        assert result["output"] == "Just plain text"


class TestLangChainTaskAgentRunnable:
    """Tests for invoke() with plain Runnable."""

    @pytest.mark.asyncio
    async def test_invoke_runnable_with_json(self):
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content='{"status": "ok"}'))

        agent = LangChainTaskAgent(mock_runnable)
        request = AgentRequest(input={"task": "Process data"})

        result = await agent.invoke(request)

        mock_runnable.ainvoke.assert_called_once_with("Process data")
        assert result["output"] == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_invoke_runnable_with_dict(self):
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value={"result": "success", "count": 3})

        agent = LangChainTaskAgent(mock_runnable)
        request = AgentRequest(input={"task": "Count items"})

        result = await agent.invoke(request)

        assert result["output"] == {"result": "success", "count": 3}
