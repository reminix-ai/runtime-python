"""Tests for the LangChain thread agent."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import Runnable

from reminix_langchain import LangChainThreadAgent
from reminix_runtime import AGENT_TYPES, AgentRequest


class TestLangChainThreadAgent:
    """Tests for the LangChainThreadAgent class."""

    def test_instantiation_with_runnable(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainThreadAgent(mock_runnable)
        assert isinstance(agent, LangChainThreadAgent)

    def test_instantiation_with_graph(self):
        mock_graph = MagicMock(spec=Runnable)
        mock_graph.nodes = {}
        agent = LangChainThreadAgent(mock_graph)
        assert isinstance(agent, LangChainThreadAgent)

    def test_custom_name(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainThreadAgent(mock_runnable, name="my-thread")
        assert agent.name == "my-thread"

    def test_default_name(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainThreadAgent(mock_runnable)
        assert agent.name == "langchain-thread-agent"

    def test_thread_type_metadata(self):
        mock_runnable = MagicMock(spec=Runnable)
        agent = LangChainThreadAgent(mock_runnable)
        assert agent.metadata["type"] == "thread"
        assert agent.metadata["inputSchema"] == AGENT_TYPES["thread"]["inputSchema"]
        assert agent.metadata["outputSchema"] == AGENT_TYPES["thread"]["outputSchema"]


class TestLangChainThreadAgentGraph:
    """Tests for invoke() with CompiledStateGraph."""

    @pytest.mark.asyncio
    async def test_invoke_graph_returns_messages(self):
        mock_graph = MagicMock(spec=Runnable)
        mock_graph.nodes = {}
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Hello"),
                    AIMessage(content="Hi there!"),
                ]
            }
        )

        agent = LangChainThreadAgent(mock_graph)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        result = await agent.invoke(request)
        messages = result["output"]

        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_invoke_graph_includes_tool_calls(self):
        mock_graph = MagicMock(spec=Runnable)
        mock_graph.nodes = {}
        mock_graph.ainvoke = AsyncMock(
            return_value={
                "messages": [
                    HumanMessage(content="Weather?"),
                    AIMessage(
                        content="",
                        tool_calls=[
                            {"id": "call_1", "name": "get_weather", "args": {"city": "London"}}
                        ],
                    ),
                    ToolMessage(content="22C sunny", tool_call_id="call_1"),
                    AIMessage(content="The weather in London is 22C and sunny."),
                ]
            }
        )

        agent = LangChainThreadAgent(mock_graph)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Weather?"}]})

        result = await agent.invoke(request)
        messages = result["output"]

        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        assert messages[2]["tool_call_id"] == "call_1"
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "The weather in London is 22C and sunny."


class TestLangChainThreadAgentRunnable:
    """Tests for invoke() with plain Runnable."""

    @pytest.mark.asyncio
    async def test_invoke_runnable_wraps_response(self):
        mock_runnable = MagicMock(spec=Runnable)
        mock_runnable.ainvoke = AsyncMock(return_value=AIMessage(content="Response"))

        agent = LangChainThreadAgent(mock_runnable)
        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hello"}]})

        result = await agent.invoke(request)
        messages = result["output"]

        assert len(messages) >= 2
        assert messages[-1]["role"] == "assistant"
        assert messages[-1]["content"] == "Response"
