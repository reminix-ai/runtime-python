"""MCP endpoint for Reminix Runtime — Streamable HTTP transport.

Exposes user-deployed tools via the MCP protocol so that any MCP client
(Claude Desktop, Cursor, the Reminix platform, etc.) can discover and call them.

Stateless: a fresh server context per request, no sessions.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import TextContent
from mcp.types import Tool as McpTool

from .tool import Tool
from .types import ToolRequest


def setup_mcp(tools: list[Tool]) -> StreamableHTTPSessionManager:
    """Create an MCP server and session manager for the given tools.

    Args:
        tools: List of runtime tools to expose via MCP.

    Returns:
        A StreamableHTTPSessionManager to integrate with FastAPI.
    """
    tool_map = {t.name: t for t in tools}

    server = Server("reminix-runtime")

    @server.list_tools()
    async def handle_list_tools() -> list[McpTool]:
        return [
            McpTool(
                name=t.name,
                description=t.metadata.get("description", f"Tool: {t.name}"),
                inputSchema=t.metadata.get("inputSchema", {"type": "object", "properties": {}}),
            )
            for t in tools
        ]

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: dict[str, Any] | None) -> list[TextContent]:
        tool = tool_map.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")

        request = ToolRequest(arguments=arguments or {})
        result = await tool.call(request)
        output = result.get("output", result)

        return [TextContent(type="text", text=json.dumps(output))]

    return StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
    )
