"""Tests for the LangGraph workflow adapter."""

from unittest.mock import MagicMock

import pytest
from langgraph.errors import GraphInterrupt
from langgraph.types import Command, Interrupt

from reminix_langgraph import LangGraphWorkflowAgent
from reminix_runtime import AGENT_TEMPLATES, AgentRequest


class TestLangGraphWorkflowAgent:
    """Tests for LangGraphWorkflowAgent instantiation and metadata."""

    def test_instantiation(self):
        """LangGraphWorkflowAgent should be instantiable."""
        mock_graph = MagicMock()
        agent = LangGraphWorkflowAgent(mock_graph)

        assert isinstance(agent, LangGraphWorkflowAgent)

    def test_default_name(self):
        """LangGraphWorkflowAgent should use default name if not provided."""
        mock_graph = MagicMock()
        agent = LangGraphWorkflowAgent(mock_graph)

        assert agent.name == "langgraph-workflow-agent"

    def test_custom_name(self):
        """LangGraphWorkflowAgent should accept a custom name."""
        mock_graph = MagicMock()
        agent = LangGraphWorkflowAgent(mock_graph, name="my-workflow")

        assert agent.name == "my-workflow"

    def test_workflow_template_metadata(self):
        """LangGraphWorkflowAgent should have workflow template metadata."""
        mock_graph = MagicMock()
        agent = LangGraphWorkflowAgent(mock_graph)

        assert agent.metadata["template"] == "workflow"
        assert agent.metadata["input"] == AGENT_TEMPLATES["workflow"]["input"]
        assert agent.metadata["output"] == AGENT_TEMPLATES["workflow"]["output"]
        assert agent.metadata["adapter"] == "langgraph"


async def _async_iter(items):
    """Helper to create an async iterator from a list."""
    for item in items:
        yield item


class TestLangGraphWorkflowAgentInvoke:
    """Tests for the invoke() method."""

    @pytest.mark.asyncio
    async def test_simple_workflow(self):
        """invoke() should collect streamed nodes as completed steps."""
        mock_graph = MagicMock()
        mock_graph.astream = lambda *_args, **_kwargs: _async_iter(
            [
                {"fetch_data": {"records": 10}},
                {"process": {"summary": "done"}},
            ]
        )

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(input={"task": "process data"})

        response = await agent.invoke(request)
        output = response["output"]

        assert output["status"] == "completed"
        assert len(output["steps"]) == 2
        assert output["steps"][0] == {
            "name": "fetch_data",
            "status": "completed",
            "output": {"records": 10},
        }
        assert output["steps"][1] == {
            "name": "process",
            "status": "completed",
            "output": {"summary": "done"},
        }
        assert output["result"] == {"summary": "done"}

    @pytest.mark.asyncio
    async def test_graph_interrupt_with_string(self):
        """invoke() should handle GraphInterrupt with string value."""

        async def _interrupt_stream(*args, **kwargs):
            yield {"step1": {"data": "partial"}}
            exc = GraphInterrupt(
                interrupts=[Interrupt(value="Please provide approval", resumable=True)]
            )
            raise exc

        mock_graph = MagicMock()
        mock_graph.astream = _interrupt_stream

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(input={"task": "approval flow"})

        response = await agent.invoke(request)
        output = response["output"]

        assert output["status"] == "paused"
        assert len(output["steps"]) == 1
        assert output["steps"][0]["status"] == "paused"
        assert output["pendingAction"]["step"] == "step1"
        assert output["pendingAction"]["type"] == "input"
        assert output["pendingAction"]["message"] == "Please provide approval"

    @pytest.mark.asyncio
    async def test_graph_interrupt_with_dict(self):
        """invoke() should handle GraphInterrupt with dict value containing type/message."""

        async def _interrupt_stream(*args, **kwargs):
            yield {"validate": {"ok": True}}
            exc = GraphInterrupt(
                interrupts=[
                    Interrupt(
                        value={
                            "type": "approval",
                            "message": "Approve deployment?",
                            "options": ["yes", "no"],
                        },
                        resumable=True,
                    )
                ]
            )
            raise exc

        mock_graph = MagicMock()
        mock_graph.astream = _interrupt_stream

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(input={"task": "deploy"})

        response = await agent.invoke(request)
        output = response["output"]

        assert output["status"] == "paused"
        assert output["pendingAction"]["type"] == "approval"
        assert output["pendingAction"]["message"] == "Approve deployment?"
        assert output["pendingAction"]["options"] == ["yes", "no"]

    @pytest.mark.asyncio
    async def test_graph_interrupt_with_other_value(self):
        """invoke() should handle GraphInterrupt with non-string non-dict value."""

        async def _interrupt_stream(*args, **kwargs):
            exc = GraphInterrupt(interrupts=[Interrupt(value=42, resumable=True)])
            raise exc
            yield  # pragma: no cover — makes this an async generator

        mock_graph = MagicMock()
        mock_graph.astream = _interrupt_stream

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(input={"task": "test"})

        response = await agent.invoke(request)
        output = response["output"]

        assert output["status"] == "paused"
        assert output["pendingAction"]["type"] == "input"
        assert output["pendingAction"]["message"] == "42"

    @pytest.mark.asyncio
    async def test_resume(self):
        """invoke() should pass Command(resume=...) when request.input has resume."""
        captured_inputs = []

        async def _capture_stream(input_val, config):
            captured_inputs.append(input_val)
            yield {"final": {"result": "approved"}}

        mock_graph = MagicMock()
        mock_graph.astream = _capture_stream

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(
            input={"task": "deploy", "resume": {"step": "approve", "input": {"approved": True}}},
        )

        response = await agent.invoke(request)

        assert len(captured_inputs) == 1
        assert isinstance(captured_inputs[0], Command)
        assert response["output"]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """invoke() should set status=failed and mark last step on error."""

        async def _error_stream(*args, **kwargs):
            yield {"step1": {"partial": True}}
            raise RuntimeError("Graph execution failed")

        mock_graph = MagicMock()
        mock_graph.astream = _error_stream

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(input={"task": "failing task"})

        response = await agent.invoke(request)
        output = response["output"]

        assert output["status"] == "failed"
        assert len(output["steps"]) == 1
        assert output["steps"][0]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_thread_id_in_context(self):
        """invoke() should pass thread_id from context to graph config."""
        captured_configs = []

        async def _capture_stream(input_val, config):
            captured_configs.append(config)
            yield {"node1": {"ok": True}}

        mock_graph = MagicMock()
        mock_graph.astream = _capture_stream

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(
            input={"task": "test"},
            context={"thread_id": "thread-123"},
        )

        await agent.invoke(request)

        assert len(captured_configs) == 1
        assert captured_configs[0] == {"configurable": {"thread_id": "thread-123"}}

    @pytest.mark.asyncio
    async def test_no_context(self):
        """invoke() should pass empty config when no context is provided."""
        captured_configs = []

        async def _capture_stream(input_val, config):
            captured_configs.append(config)
            yield {"node1": {"ok": True}}

        mock_graph = MagicMock()
        mock_graph.astream = _capture_stream

        agent = LangGraphWorkflowAgent(mock_graph)
        request = AgentRequest(input={"task": "test"})

        await agent.invoke(request)

        assert len(captured_configs) == 1
        assert captured_configs[0] == {}
