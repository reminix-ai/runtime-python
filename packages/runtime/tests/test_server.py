"""Tests for the serve() function and server endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from reminix_runtime import (
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
    __version__,
)
from reminix_runtime.server import create_app


class MockAdapter(BaseAdapter):
    """A mock adapter for testing."""

    adapter_name = "mock"

    def __init__(self, name: str = "mock-agent"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        task = request.input.get("task", "unknown")
        return InvokeResponse(output=f"Completed task: {task}")

    async def chat(self, request: ChatRequest) -> ChatResponse:
        user_message = request.messages[-1].content
        response_content = f"Chat response to: {user_message}"
        return ChatResponse(
            output=response_content,
            messages=[
                *[{"role": m.role, "content": m.content} for m in request.messages],
                {"role": "assistant", "content": response_content},
            ],
        )


class TestCreateApp:
    """Tests for create_app()."""

    def test_create_app_returns_fastapi_app(self):
        """create_app should return a FastAPI application."""
        app = create_app([MockAdapter()])
        # FastAPI apps have a 'routes' attribute
        assert hasattr(app, "routes")

    def test_create_app_with_empty_agents_raises(self):
        """create_app should raise if no agents provided."""
        with pytest.raises(ValueError, match="At least one agent is required"):
            create_app([])


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """GET /health should return 200 OK."""
        app = create_app([MockAdapter()])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestInfoEndpoint:
    """Tests for the discovery endpoint."""

    @pytest.mark.asyncio
    async def test_info_endpoint(self):
        """GET /info should return runtime info and agents."""
        app = create_app([MockAdapter("agent-one"), MockAdapter("agent-two")])
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
        assert data["agents"][0]["type"] == "adapter"
        assert data["agents"][0]["adapter"] == "mock"
        assert data["agents"][0]["invoke"]["streaming"] is True
        assert data["agents"][0]["chat"]["streaming"] is True


class TestInvokeEndpoint:
    """Tests for the invoke endpoint."""

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """POST /agents/{agent}/invoke should return invoke response."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
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
        app = create_app([MockAdapter("my-agent")])
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
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/unknown-agent/invoke",
                json={"input": {"task": "test"}},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invoke_invalid_request_returns_422(self):
        """POST /agents/{agent}/invoke should return 422 for invalid request."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/my-agent/invoke",
                json={"input": {}},  # Empty input not allowed
            )

        assert response.status_code == 422


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """POST /agents/{agent}/chat should return chat response."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/my-agent/chat",
                json={"messages": [{"role": "user", "content": "hi there"}]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["output"] == "Chat response to: hi there"
        assert len(data["messages"]) == 2  # user message + assistant response

    @pytest.mark.asyncio
    async def test_chat_unknown_agent_returns_404(self):
        """POST /agents/{agent}/chat should return 404 for unknown agent."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/unknown-agent/chat",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 404
