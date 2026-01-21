"""Tests for AgentAdapter."""

import pytest

from reminix_runtime import AgentAdapter, ExecuteRequest, ExecuteResponse


class TestAgentAdapterContract:
    """Tests for the AgentAdapter contract."""

    def test_cannot_instantiate_adapter_base(self):
        """AgentAdapter is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AgentAdapter()  # type: ignore

    def test_subclass_must_implement_name(self):
        """Subclass must implement the name property."""

        class IncompleteAdapter(AgentAdapter):
            async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
                return ExecuteResponse(output="")

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_subclass_must_implement_execute(self):
        """Subclass must implement the execute method."""

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

            async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
                # Check if it's a chat-style request (has messages)
                if "messages" in request.input:
                    user_msg = request.input["messages"][-1]["content"]
                    return {"output": f"Hello from chat: {user_msg}"}
                # Otherwise, it's an invoke-style request
                task = request.input.get("task", "unknown")
                return {"output": f"Completed: {task}"}

        return TestAdapter()

    def test_adapter_has_name(self):
        """Adapter should have a name property."""
        adapter = self._create_adapter()
        assert adapter.name == "test-agent"

    @pytest.mark.asyncio
    async def test_execute_returns_response(self):
        """Execute should return an ExecuteResponse (dict)."""
        adapter = self._create_adapter()
        request = ExecuteRequest(input={"task": "summarize"})
        response = await adapter.execute(request)

        assert isinstance(response, dict)
        assert response["output"] == "Completed: summarize"

    @pytest.mark.asyncio
    async def test_execute_with_messages_returns_response(self):
        """Execute with messages input should return an ExecuteResponse (dict)."""
        adapter = self._create_adapter()
        request = ExecuteRequest(input={"messages": [{"role": "user", "content": "hello"}]})
        response = await adapter.execute(request)

        assert isinstance(response, dict)
        assert response["output"] == "Hello from chat: hello"

    @pytest.mark.asyncio
    async def test_execute_stream_not_implemented_by_default(self):
        """execute_stream should raise NotImplementedError by default."""
        adapter = self._create_adapter()
        request = ExecuteRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError):
            async for _ in adapter.execute_stream(request):
                pass
