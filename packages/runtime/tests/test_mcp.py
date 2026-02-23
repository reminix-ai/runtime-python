"""Tests for the MCP endpoint (POST /mcp)."""

import json
from typing import Any

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from reminix_runtime import tool
from reminix_runtime.server import create_app


async def mcp_request(
    client: AsyncClient,
    method: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send a JSON-RPC message to POST /mcp and return the parsed response."""
    message: dict[str, Any] = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
    }
    if params is not None:
        message["params"] = params

    response = await client.post(
        "/mcp",
        json=message,
        headers={"Accept": "application/json, text/event-stream"},
    )

    assert response.status_code == 200
    return response.json()


@tool
async def greet(name: str) -> dict:
    """Greet someone by name."""
    return {"message": f"Hello, {name}!"}


@tool
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


class TestMcpInitialize:
    """Tests for MCP initialize."""

    @pytest.mark.asyncio
    async def test_initialize_returns_capabilities(self):
        """POST /mcp with initialize should return server capabilities."""
        app = create_app(tools=[greet])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await mcp_request(
                    client,
                    "initialize",
                    {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "test", "version": "1.0.0"},
                    },
                )

        assert "result" in response
        result = response["result"]
        assert result["protocolVersion"] == "2025-03-26"
        assert result["serverInfo"]["name"] == "reminix-runtime"


class TestMcpToolsList:
    """Tests for MCP tools/list."""

    @pytest.mark.asyncio
    async def test_list_tools(self):
        """POST /mcp with tools/list should return all registered tools."""
        app = create_app(tools=[greet, add])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await mcp_request(client, "tools/list")

        assert "result" in response
        tools = response["result"]["tools"]
        assert len(tools) == 2

        names = [t["name"] for t in tools]
        assert "greet" in names
        assert "add" in names

        greet_tool = next(t for t in tools if t["name"] == "greet")
        assert greet_tool["description"] == "Greet someone by name."

    @pytest.mark.asyncio
    async def test_list_tools_includes_input_schema(self):
        """tools/list should include inputSchema for each tool."""
        app = create_app(tools=[greet])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await mcp_request(client, "tools/list")

        tools = response["result"]["tools"]
        greet_tool = tools[0]
        assert "inputSchema" in greet_tool
        assert greet_tool["inputSchema"]["type"] == "object"
        assert "properties" in greet_tool["inputSchema"]


class TestMcpToolsCall:
    """Tests for MCP tools/call."""

    @pytest.mark.asyncio
    async def test_call_tool(self):
        """POST /mcp with tools/call should execute the tool."""
        app = create_app(tools=[greet])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await mcp_request(
                    client,
                    "tools/call",
                    {
                        "name": "greet",
                        "arguments": {"name": "World"},
                    },
                )

        assert "result" in response
        content = response["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"

        output = json.loads(content[0]["text"])
        assert output == {"message": "Hello, World!"}

    @pytest.mark.asyncio
    async def test_call_tool_numeric_result(self):
        """tools/call should handle numeric return values."""
        app = create_app(tools=[add])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await mcp_request(
                    client,
                    "tools/call",
                    {
                        "name": "add",
                        "arguments": {"a": 3, "b": 7},
                    },
                )

        content = response["result"]["content"]
        output = json.loads(content[0]["text"])
        assert output == 10

    @pytest.mark.asyncio
    async def test_call_tool_error(self):
        """tools/call should handle tool execution errors."""

        @tool
        async def failing(param: str) -> dict:
            """A tool that always fails."""
            raise ValueError("Something went wrong")

        app = create_app(tools=[failing])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await mcp_request(
                    client,
                    "tools/call",
                    {
                        "name": "failing",
                        "arguments": {"param": "test"},
                    },
                )

        result = response["result"]
        assert result.get("isError") is True


class TestMcpHttpMethods:
    """Tests for MCP HTTP methods."""

    @pytest.mark.asyncio
    async def test_get_mcp_not_supported(self):
        """GET /mcp should not succeed (no SSE in stateless mode)."""
        app = create_app(tools=[greet])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await client.get("/mcp")

        # MCP SDK returns 406 (no acceptable SSE stream in stateless mode)
        assert response.status_code in (405, 406)

    @pytest.mark.asyncio
    async def test_delete_mcp_returns_405(self):
        """DELETE /mcp should return 405 in stateless mode."""
        app = create_app(tools=[greet])
        async with LifespanManager(app) as manager:
            async with AsyncClient(
                transport=ASGITransport(app=manager.app), base_url="http://test"
            ) as client:
                response = await client.delete("/mcp")

        assert response.status_code == 405
