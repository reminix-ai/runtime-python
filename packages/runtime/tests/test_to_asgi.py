"""Tests for the to_asgi() method."""

import json

import pytest

from reminix_runtime import Agent, AgentInvokeRequest, AgentInvokeResponse


async def call_asgi(app, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
    """Helper to call an ASGI app and return status and JSON body."""
    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "headers": [[b"content-type", b"application/json"]],
    }

    body_bytes = json.dumps(body).encode() if body else b""
    body_sent = False

    async def receive():
        nonlocal body_sent
        if not body_sent:
            body_sent = True
            return {"type": "http.request", "body": body_bytes, "more_body": False}
        return {"type": "http.disconnect"}

    response_started = False
    status = 0
    response_body = b""

    async def send(message):
        nonlocal response_started, status, response_body
        if message["type"] == "http.response.start":
            response_started = True
            status = message["status"]
        elif message["type"] == "http.response.body":
            response_body += message.get("body", b"")

    await app(scope, receive, send)

    if response_body:
        return status, json.loads(response_body)
    return status, {}


class TestToAsgi:
    """Tests for Agent.to_asgi() method."""

    def test_to_asgi_returns_callable(self):
        """to_asgi() returns a callable ASGI app."""
        agent = Agent("test-agent")
        app = agent.to_asgi()
        assert callable(app)

    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """ASGI app handles /health endpoint."""
        agent = Agent("test-agent")
        app = agent.to_asgi()

        status, body = await call_asgi(app, "GET", "/health")

        assert status == 200
        assert body == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_info_endpoint(self):
        """ASGI app handles /info endpoint."""
        agent = Agent("test-agent", metadata={"version": "1.0"})
        app = agent.to_asgi()

        status, body = await call_asgi(app, "GET", "/info")

        assert status == 200
        assert body["runtime"]["name"] == "reminix-runtime"
        assert len(body["agents"]) == 1
        assert body["agents"][0]["name"] == "test-agent"
        assert body["agents"][0]["version"] == "1.0"

    @pytest.mark.asyncio
    async def test_invoke_endpoint(self):
        """ASGI app handles /agents/{name}/invoke endpoint with input wrapper."""
        test_agent = Agent("test-agent")

        @test_agent.handler
        async def handle(request: AgentInvokeRequest) -> AgentInvokeResponse:
            prompt = request.input.get("prompt", "")
            return {"output": f"Received: {prompt}"}

        app = test_agent.to_asgi()

        # New API uses { input: { ... } }
        status, body = await call_asgi(
            app, "POST", "/agents/test-agent/invoke", {"input": {"prompt": "hello"}}
        )

        assert status == 200
        assert body["output"] == "Received: hello"

    @pytest.mark.asyncio
    async def test_wrong_agent_name_returns_404(self):
        """ASGI app returns 404 for wrong agent name."""
        test_agent = Agent("test-agent")

        @test_agent.handler
        async def handle(request: AgentInvokeRequest) -> AgentInvokeResponse:
            return {"output": "ok"}

        app = test_agent.to_asgi()

        status, body = await call_asgi(
            app, "POST", "/agents/wrong-agent/invoke", {"input": {"prompt": "test"}}
        )

        assert status == 404
        assert "not found" in body["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self):
        """ASGI app returns 404 for unknown path."""
        agent = Agent("test-agent")
        app = agent.to_asgi()

        status, body = await call_asgi(app, "GET", "/unknown")

        assert status == 404

    @pytest.mark.asyncio
    async def test_cors_preflight(self):
        """ASGI app handles CORS preflight."""
        agent = Agent("test-agent")
        app = agent.to_asgi()

        scope = {
            "type": "http",
            "method": "OPTIONS",
            "path": "/health",
            "headers": [],
        }

        response_started = False
        status = 0
        headers = []

        async def receive():
            return {"type": "http.disconnect"}

        async def send(message):
            nonlocal response_started, status, headers
            if message["type"] == "http.response.start":
                response_started = True
                status = message["status"]
                headers = message.get("headers", [])

        await app(scope, receive, send)

        assert status == 204
        # Check CORS headers
        header_dict = {k.decode(): v.decode() for k, v in headers}
        assert header_dict.get("access-control-allow-origin") == "*"
