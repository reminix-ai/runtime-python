"""Tests for the serve() function and server endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from reminix_runtime import (
    BaseAdapter,
    InvokeRequest,
    InvokeResponse,
    ChatRequest,
    ChatResponse,
)
from reminix_runtime.server import create_app


class MockAdapter(BaseAdapter):
    """A mock adapter for testing."""

    def __init__(self, name: str = "mock-agent"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        user_message = request.messages[-1].content
        return InvokeResponse(
            content=f"Invoked with: {user_message}",
            messages=[{"role": "assistant", "content": f"Invoked with: {user_message}"}],
        )

    async def chat(self, request: ChatRequest) -> ChatResponse:
        user_message = request.messages[-1].content
        return ChatResponse(
            content=f"Chat response to: {user_message}",
            messages=[{"role": "assistant", "content": f"Chat response to: {user_message}"}],
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
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAgentsEndpoint:
    """Tests for listing available agents."""

    @pytest.mark.asyncio
    async def test_list_agents(self):
        """GET /agents should return list of agent names."""
        app = create_app([MockAdapter("agent-one"), MockAdapter("agent-two")])
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/agents")

        assert response.status_code == 200
        data = response.json()
        assert data["agents"] == ["agent-one", "agent-two"]


class TestInvokeEndpoint:
    """Tests for the invoke endpoint."""

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """POST /agents/{agent}/invoke should return invoke response."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/agents/my-agent/invoke",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Invoked with: hello"
        assert len(data["messages"]) == 1

    @pytest.mark.asyncio
    async def test_invoke_with_context(self):
        """POST /agents/{agent}/invoke should accept context."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/agents/my-agent/invoke",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "context": {"user_id": "123"},
                },
            )

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_invoke_unknown_agent_returns_404(self):
        """POST /agents/{agent}/invoke should return 404 for unknown agent."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/agents/unknown-agent/invoke",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_invoke_invalid_request_returns_422(self):
        """POST /agents/{agent}/invoke should return 422 for invalid request."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/agents/my-agent/invoke",
                json={"messages": []},  # Empty messages not allowed
            )

        assert response.status_code == 422


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    @pytest.mark.asyncio
    async def test_chat_success(self):
        """POST /agents/{agent}/chat should return chat response."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/agents/my-agent/chat",
                json={"messages": [{"role": "user", "content": "hi there"}]},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["content"] == "Chat response to: hi there"
        assert len(data["messages"]) == 1

    @pytest.mark.asyncio
    async def test_chat_unknown_agent_returns_404(self):
        """POST /agents/{agent}/chat should return 404 for unknown agent."""
        app = create_app([MockAdapter("my-agent")])
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/agents/unknown-agent/chat",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        assert response.status_code == 404
