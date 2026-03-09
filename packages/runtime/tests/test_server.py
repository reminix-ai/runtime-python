"""Tests for the serve() function and server endpoints."""

from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient

from reminix_runtime import (
    Agent,
    AgentRequest,
    __version__,
    tool,
)
from reminix_runtime.server import create_app


class MockTaskAgent(Agent):
    """A mock agent for testing task-style requests."""

    def __init__(self, name: str = "mock-agent"):
        super().__init__(
            name=name,
            streaming=True,
            framework="mock",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        task = request.input.get("task", "unknown")
        return {"output": f"Completed task: {task}"}

    async def invoke_stream(self, request: AgentRequest):
        yield ""


class MockChatAgent(Agent):
    """A mock agent for testing chat-style requests."""

    def __init__(self, name: str = "mock-agent"):
        super().__init__(
            name=name,
            streaming=True,
            framework="mock",
            input_schema={"type": "object"},
            output_schema={"type": "string"},
        )

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        messages = request.input.get("messages", [])
        user_message = messages[-1]["content"] if messages else ""
        return {
            "output": {
                "messages": [{"role": "assistant", "content": f"Chat response to: {user_message}"}]
            }
        }

    async def invoke_stream(self, request: AgentRequest):
        yield ""


class TestCreateApp:
    """Tests for create_app()."""

    def test_create_app_returns_fastapi_app(self):
        """create_app should return a FastAPI application."""
        app = create_app(agents=[MockTaskAgent()])
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
        app = create_app(agents=[MockTaskAgent()])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestManifestEndpoint:
    """Tests for the discovery endpoint."""

    @pytest.mark.asyncio
    async def test_manifest_endpoint(self):
        """GET /manifest should return runtime info and endpoints."""
        app = create_app(agents=[MockTaskAgent("agent-one"), MockTaskAgent("agent-two")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/manifest")

        assert response.status_code == 200
        data = response.json()

        # Check runtime info
        assert data["runtime"]["name"] == "reminix-runtime"
        assert data["runtime"]["version"] == __version__
        assert data["runtime"]["language"] == "python"

        # Check endpoints
        assert len(data["endpoints"]) == 2
        assert data["endpoints"][0]["kind"] == "agent"
        assert data["endpoints"][0]["name"] == "agent-one"
        assert data["endpoints"][0]["path"] == "/agents/agent-one/invoke"
        assert data["endpoints"][0]["framework"] == "mock"
        assert data["endpoints"][0]["capabilities"]["streaming"] is True
        assert data["endpoints"][1]["name"] == "agent-two"


class TestInvokeEndpoint:
    """Tests for the invoke endpoint."""

    @pytest.mark.asyncio
    async def test_invoke_success(self):
        """POST /agents/{agent}/invoke should return invoke response."""
        app = create_app(agents=[MockTaskAgent("my-agent")])
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
        app = create_app(agents=[MockTaskAgent("my-agent")])
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
        app = create_app(agents=[MockTaskAgent("my-agent")])
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
        app = create_app(agents=[MockChatAgent("my-agent")])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/my-agent/invoke",
                json={"input": {"messages": [{"role": "user", "content": "hi there"}]}},
            )

        assert response.status_code == 200
        data = response.json()
        assert data["output"]["messages"][0]["role"] == "assistant"
        assert data["output"]["messages"][0]["content"] == "Chat response to: hi there"


class TestStreamingEndpoint:
    """Tests for the streaming SSE endpoint."""

    @pytest.mark.asyncio
    async def test_stream_typed_events_and_done(self):
        """Streaming should return typed text_delta events and [DONE]."""

        class StreamAgent(Agent):
            def __init__(self):
                super().__init__("stream-agent", streaming=True)

            async def invoke(self, request):
                return {"output": "ok"}

            async def invoke_stream(self, request):
                yield "Hello "
                yield "world"

        app = create_app(agents=[StreamAgent()])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/stream-agent/invoke",
                json={"input": {"task": "test"}, "stream": True},
            )

        assert response.status_code == 200
        text = response.text

        # Should contain typed text_delta events
        assert '"type": "text_delta"' in text or '"type":"text_delta"' in text
        assert "Hello " in text
        assert "world" in text
        # Should end with [DONE]
        assert "[DONE]" in text

    @pytest.mark.asyncio
    async def test_stream_event_objects(self):
        """Streaming should handle StreamEvent objects directly."""
        from reminix_runtime.stream_events import MessageEvent
        from reminix_runtime.types import Message

        class EventAgent(Agent):
            def __init__(self):
                super().__init__("event-agent", streaming=True)

            async def invoke(self, request):
                return {"output": "ok"}

            async def invoke_stream(self, request):
                yield MessageEvent(message=Message(role="assistant", content="Hi"))

        app = create_app(agents=[EventAgent()])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/event-agent/invoke",
                json={"input": {}, "stream": True},
            )

        assert response.status_code == 200
        text = response.text
        assert "message" in text
        assert "assistant" in text

    @pytest.mark.asyncio
    async def test_stream_errors_as_event_error(self):
        """Streaming errors should be sent as event: error."""

        class ErrorAgent(Agent):
            def __init__(self):
                super().__init__("error-agent", streaming=True)

            async def invoke(self, request):
                return {"output": "ok"}

            async def invoke_stream(self, request):
                yield "partial"
                raise RuntimeError("Stream failed")

        app = create_app(agents=[ErrorAgent()])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/error-agent/invoke",
                json={"input": {}, "stream": True},
            )

        assert response.status_code == 200
        text = response.text
        assert "event: error" in text
        assert "Stream failed" in text

    @pytest.mark.asyncio
    async def test_stream_returns_501_for_non_streaming_agent(self):
        """Streaming request to non-streaming agent should return 501."""

        class NonStreamingAgent(Agent):
            def __init__(self):
                super().__init__("no-stream", streaming=False)

            async def invoke(self, request):
                return {"output": "ok"}

        app = create_app(agents=[NonStreamingAgent()])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/agents/no-stream/invoke",
                json={"input": {}, "stream": True},
            )

        assert response.status_code == 501
