"""Reminix Runtime Server."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import __version__
from .agent import AgentBase
from .tool import ToolBase
from .types import (
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
    ToolExecuteRequest,
    ToolExecuteResponse,
)


async def _sse_generator(stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Convert an async string iterator to SSE format."""
    try:
        async for chunk in stream:
            yield f"data: {chunk}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except NotImplementedError as e:
        yield f'data: {{"error": "{str(e)}"}}\n\n'.encode()


def create_app(
    *,
    agents: list[AgentBase] | None = None,
    tools: list[ToolBase] | None = None,
) -> FastAPI:
    """Create a FastAPI application with agent and tool endpoints.

    Args:
        agents: List of agents.
        tools: List of tools.

    Returns:
        A FastAPI application instance.

    Raises:
        ValueError: If no agents or tools are provided.
    """
    agents = agents or []
    tools = tools or []

    if not agents and not tools:
        raise ValueError("At least one agent or tool is required")

    # Build lookup dicts by name
    agent_map: dict[str, AgentBase] = {agent.name: agent for agent in agents}
    tool_map: dict[str, ToolBase] = {tool.name: tool for tool in tools}

    app = FastAPI(title="Reminix Runtime")

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/info")
    async def info() -> dict[str, Any]:
        """Runtime discovery endpoint."""
        return {
            "runtime": {
                "name": "reminix-runtime",
                "version": __version__,
                "language": "python",
                "framework": "fastapi",
            },
            "agents": [
                {
                    "name": agent.name,
                    **agent.metadata,
                    "invoke": {"streaming": agent.invoke_streaming},
                    "chat": {"streaming": agent.chat_streaming},
                }
                for agent in agents
            ],
            "tools": [
                {
                    "name": tool.name,
                    **tool.metadata,
                }
                for tool in tools
            ],
        }

    @app.post("/agents/{agent_name}/invoke", response_model=None)
    async def invoke(agent_name: str, request: InvokeRequest) -> InvokeResponse | StreamingResponse:
        """Invoke an agent."""
        agent = agent_map.get(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

        if request.stream:
            return StreamingResponse(
                _sse_generator(agent.invoke_stream(request)),
                media_type="text/event-stream",
            )

        return await agent.invoke(request)

    @app.post("/agents/{agent_name}/chat", response_model=None)
    async def chat(agent_name: str, request: ChatRequest) -> ChatResponse | StreamingResponse:
        """Chat with an agent."""
        agent = agent_map.get(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

        if request.stream:
            return StreamingResponse(
                _sse_generator(agent.chat_stream(request)),
                media_type="text/event-stream",
            )

        return await agent.chat(request)

    @app.post("/tools/{tool_name}/execute", response_model=None)
    async def execute_tool(tool_name: str, request: ToolExecuteRequest) -> ToolExecuteResponse:
        """Execute a tool."""
        tool = tool_map.get(tool_name)
        if tool is None:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

        return await tool.execute(request)

    return app


def serve(
    *,
    agents: list[AgentBase] | None = None,
    tools: list[ToolBase] | None = None,
    port: int | None = None,
    host: str | None = None,
) -> None:
    """Serve agents and tools via REST API.

    Args:
        agents: List of agents.
        tools: List of tools.
        port: Port to serve on. Defaults to PORT environment variable or 8080.
        host: Host to bind to. Defaults to "0.0.0.0" (all interfaces, IPv4 and IPv6).
            Can be overridden via HOST environment variable. Set to "::" for IPv6-only.
    """
    import os

    import uvicorn

    # Allow override via environment variable (useful for Fly deployments)
    if host is None:
        host = os.getenv("HOST", "0.0.0.0")

    if port is None:
        port = int(os.getenv("PORT", "8080"))

    app = create_app(agents=agents, tools=tools)
    uvicorn.run(app, host=host, port=port)
