"""Tests for AgentAdapter."""

import pytest

from reminix_runtime import AgentAdapter, AgentInvokeRequest, AgentInvokeResponse


class TestAgentAdapterContract:
    """Tests for the AgentAdapter contract."""

    def test_cannot_instantiate_adapter_base(self):
        """AgentAdapter is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentAdapter()  # type: ignore

    def test_subclass_must_implement_name(self):
        """Subclass must implement the name property."""

        class IncompleteAdapter(AgentAdapter):
            async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponse:
                return AgentInvokeResponse(output="")

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_subclass_must_implement_invoke(self):
        """Subclass must implement the invoke method."""

        class IncompleteAdapter(AgentAdapter):
            @property
            def name(self) -> str:
                return "test"

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore


class TestConcreteAdapter:
    """Tests for a concrete adapter implementation."""

    def _create_adapter(self) -> AgentAdapter:
        """Create a minimal concrete adapter for testing."""

        class TestAdapter(AgentAdapter):
            @property
            def name(self) -> str:
                return "test-agent"

            async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponse:
                # Check if it's a chat-style request (has messages)
                if "messages" in request.input:
                    user_msg = request.input["messages"][-1]["content"]
                    return {"output": f"Hello from chat: {user_msg}"}
                # Otherwise, it's a task-style request
                task = request.input.get("task", "unknown")
                return {"output": f"Completed: {task}"}

        return TestAdapter()

    def test_adapter_has_name(self):
        """Adapter should have a name property."""
        adapter = self._create_adapter()
        assert adapter.name == "test-agent"

    @pytest.mark.asyncio
    async def test_invoke_returns_response(self):
        """Invoke should return an AgentInvokeResponse (dict)."""
        adapter = self._create_adapter()
        request = AgentInvokeRequest(input={"task": "summarize"})
        response = await adapter.invoke(request)

        assert isinstance(response, dict)
        assert response["output"] == "Completed: summarize"

    @pytest.mark.asyncio
    async def test_invoke_with_messages_returns_response(self):
        """Invoke with messages input should return an AgentInvokeResponse (dict)."""
        adapter = self._create_adapter()
        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "hello"}]})
        response = await adapter.invoke(request)

        assert isinstance(response, dict)
        assert response["output"] == "Hello from chat: hello"

    @pytest.mark.asyncio
    async def test_invoke_stream_not_implemented_by_default(self):
        """invoke_stream should raise NotImplementedError by default."""
        adapter = self._create_adapter()
        request = AgentInvokeRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError):
            async for _ in adapter.invoke_stream(request):
                pass

    def test_adapter_has_capabilities(self):
        """Adapter should have capabilities in metadata."""
        adapter = self._create_adapter()
        assert adapter.metadata["capabilities"]["streaming"] is True
