"""Tests for the serve() function and server endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from reminix_runtime import (
    AgentAdapter,
    AgentInvokeRequest,
    AgentInvokeResponse,
    __version__,
    tool,
)
from reminix_runtime.server import create_app


class MockTaskAdapter(AgentAdapter):
    """A mock adapter for testing task-style requests."""

    adapter_name = "mock"

    def __init__(self, name: str = "mock-agent"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponse:
        task = request.input.get("task", "unknown")
        return {"output": f"Completed task: {task}"}


class MockChatAdapter(AgentAdapter):
    """A mock adapter for testing chat-style requests."""

    adapter_name = "mock"

    def __init__(self, name: str = "mock-agent"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponse:
        messages = request.input.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        return {
            "output": {
                "messages": [{"role": "assistant", "content": f"Chat response to: {user_message}"}]
            }
        }


class TestCreateApp:
    """Tests for create_app()."""

    def test_create_app_returns_fastapi_app(self):
        """create_app should return a FastAPI application."""
        app = create_app(agents=[MockTaskAdapter()])
        # FastAPI apps have a 'routes' attribute
        assert hasattr(app, "routes")

    def test_create_app_with_no_agents_or_tools_raises(self):
        """create_app should raise if no agents or tools provided."""
        with pytest.raises(ValueError, match="At least one agent or tool is required"):
            create_app()

    def test_create_app_with_only_tools(self):
        """create_app should work with only tools."""

        @tool
        async def my_tool(param: str) -> dict:
            """A test tool."""
            return {"result": param}

        app = create_app(tools=[my_tool])
        assert hasattr(app, "routes")


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """GET /health should return 200 OK."""
        app = create_app(agents=[MockTaskAdapter()])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestInfoEndpoint:
    """Tests for the discovery endpoint."""

    @pytest.mark.asyncio
    async def test_info_endpoint(self):
        """GET /info should return runtime info and agents."""
        app = create_app(agents=[MockTaskAdapter("agent-one"), MockTaskAdapter("agent-two")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/info")

        assert response.status_code == 200
        data = response.json()

        # Check runtime info
        assert data["runtime"]["name"] == "reminix-runtime"
        assert data["runtime"]["version"] == __version__
        assert data["runtime"]["language"] == "python"
        assert data["runtime"]["framework"] == "fastapi"

        # Check agents
        assert len(data["agents"]) == 2
        assert data["agents"][0]["name"] == "agent-one"
        assert data["agents"][0]["adapter"] == "mock"
        assert data["agents"][0]["capabilities"]["streaming"] is True


class TestInvokeEndpoint:
    """Tests for the invoke endpoint."""

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """POST /agents/{agent}/invoke should return invoke response."""
        app = create_app(agents=[MockTaskAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            # New API uses { input: { ... } }
            response = await client.post(
                "/agents/my-agent/invoke",
                json={"input": {"task": "summarize"}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["output"] == "Completed task: summarize"

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        """POST /agents/{agent}/invoke should accept context."""
        app = create_app(agents=[MockTaskAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/my-agent/invoke",
                json={
                    "input": {"task": "test"},
                    "context": {"user_id": "123"},
                },
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invoke_unknown_agent_returns_404(self):
        """POST /agents/{agent}/invoke should return 404 for unknown agent."""
        app = create_app(agents=[MockTaskAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/unknown-agent/invoke",
                json={"input": {"task": "test"}},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_invoke_with_messages_input(self):
        """POST /agents/{agent}/invoke should handle chat-style input."""
        app = create_app(agents=[MockChatAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/my-agent/invoke",
                json={"input": {"messages": [{"role": "user", "content": "hi there"}]}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["output"]["messages"][0]["role"] == "assistant"
        assert data["output"]["messages"][0]["content"] == "Chat response to: hi there"


class TestToolCallEndpoint:
    """Tests for the tool call endpoint."""

    @pytest.mark.asyncio
    async def test_call_tool_success(self):
        """POST /tools/{tool}/call should return tool response."""

        @tool
        async def greet(name: str) -> dict:
            """Greet someone."""
            return {"message": f"Hello, {name}!"}

        app = create_app(tools=[greet])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/tools/greet/call",
                json={"input": {"name": "World"}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["output"] == {"message": "Hello, World!"}

    @pytest.mark.asyncio
    async def test_call_tool_with_context(self):
        """POST /tools/{tool}/call should accept context."""

        @tool
        async def my_tool(param: str) -> dict:
            """A test tool."""
            return {"param": param}

        app = create_app(tools=[my_tool])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/tools/my_tool/call",
                json={"input": {"param": "test"}, "context": {"user_id": "123"}},
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_call_tool_unknown_returns_404(self):
        """POST /tools/{tool}/call should return 404 for unknown tool."""

        @tool
        async def my_tool(param: str) -> dict:
            """A test tool."""
            return {"param": param}

        app = create_app(tools=[my_tool])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/tools/unknown_tool/call",
                json={"input": {"param": "test"}},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_call_tool_with_error(self):
        """POST /tools/{tool}/call should return error response on exception."""

        @tool
        async def failing_tool(param: str) -> dict:
            """A tool that fails."""
            raise ValueError("Something went wrong")

        app = create_app(tools=[failing_tool])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/tools/failing_tool/call",
                json={"input": {"param": "test"}},
            )

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert data["error"]["message"] == "Something went wrong"
        assert data["error"]["type"] == "ValidationError"

    @pytest.mark.asyncio
    async def test_call_sync_tool(self):
        """POST /tools/{tool}/call should work with sync tools."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        app = create_app(tools=[add])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/tools/add/call",
                json={"input": {"a": 2, "b": 3}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["output"] == 5


class TestInfoEndpointWithTools:
    """Tests for the info endpoint with tools."""

    @pytest.mark.asyncio
    async def test_info_includes_tools(self):
        """GET /info should include tools."""

        @tool
        async def my_tool(param: str) -> dict:
            """My tool description."""
            return {"param": param}

        app = create_app(tools=[my_tool])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/info")

        assert response.status_code == 200
        data = response.json()

        assert "tools" in data
        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "my_tool"
        assert data["tools"][0]["description"] == "My tool description."
        assert "input" in data["tools"][0]

    @pytest.mark.asyncio
    async def test_info_with_agents_and_tools(self):
        """GET /info should include both agents and tools."""

        @tool
        async def my_tool(param: str) -> dict:
            """A test tool."""
            return {"param": param}

        app = create_app(agents=[MockTaskAdapter("my-agent")], tools=[my_tool])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/info")

        assert response.status_code == 200
        data = response.json()

        assert len(data["agents"]) == 1
        assert data["agents"][0]["name"] == "my-agent"

        assert len(data["tools"]) == 1
        assert data["tools"][0]["name"] == "my_tool"
