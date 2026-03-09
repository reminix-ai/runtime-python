"""Reminix Runtime Server."""

import contextlib
import json
import os
import traceback
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .agent import Agent
from .mcp_endpoint import setup_mcp
from .stream_events import StreamEvent
from .tool import Tool
from .types import AgentRequest

# Enable debug mode via environment variable to include stack traces in error responses
REMINIX_CLOUD = os.getenv("REMINIX_CLOUD", "").lower() in ("true", "1", "yes")


def _create_error_response(
    error: Exception,
    error_type: str = "ExecutionError",
) -> dict[str, Any]:
    """Create a structured error response.

    Args:
        error: The exception that occurred.
        error_type: The type/category of the error.

    Returns:
        A structured error response dict.
    """
    response: dict[str, Any] = {
        "error": {
            "type": error_type,
            "message": str(error),
        }
    }

    # Include stack trace in debug mode
    if REMINIX_CLOUD:
        response["error"]["stack"] = traceback.format_exc()

    return response


def normalize_stream_chunk(chunk: str | StreamEvent) -> dict[str, Any]:
    """Normalize a stream chunk to a StreamEvent dict.

    Raw strings are wrapped as text_delta events; StreamEvent objects are serialized.
    """
    if isinstance(chunk, str):
        return {"type": "text_delta", "delta": chunk}
    return chunk.model_dump(exclude_none=True)


async def _sse_generator(stream: AsyncIterator[str | StreamEvent]) -> AsyncIterator[bytes]:
    """Convert an async stream of strings/StreamEvents to typed SSE format."""
    try:
        async for chunk in stream:
            event = normalize_stream_chunk(chunk)
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except NotImplementedError as e:
        error_data = _create_error_response(e, "NotImplementedError")
        yield f"event: error\ndata: {json.dumps(error_data['error'])}\n\n".encode()
    except Exception as e:
        error_data = _create_error_response(e, type(e).__name__)
        yield f"event: error\ndata: {json.dumps(error_data['error'])}\n\n".encode()


def create_app(
    *,
    agents: list[Agent] | None = None,
    tools: list[Tool] | None = None,
) -> FastAPI:
    """Create a FastAPI application with agent and tool endpoints.

    Args:
        agents: List of agents.
        tools: List of tools.

    Returns:
        A FastAPI application instance.

    Raises:
        ValueError: If no agents or tools are provided.
        ValueError: If duplicate agent or tool names are found.
    """
    agents = agents or []
    tools = tools or []

    if not agents and not tools:
        raise ValueError("At least one agent or tool is required")

    # Build lookup dicts by name (with duplicate detection)
    agent_map: dict[str, Agent] = {}
    for a in agents:
        if a.name in agent_map:
            raise ValueError(f"Duplicate agent name: '{a.name}'")
        agent_map[a.name] = a

    tool_map: dict[str, Tool] = {}
    for t in tools:
        if t.name in tool_map:
            raise ValueError(f"Duplicate tool name: '{t.name}'")
        tool_map[t.name] = t

    # Set up MCP session manager for tool discovery and execution
    mcp_session_manager = setup_mcp(tools)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with mcp_session_manager.run():
            yield

    app = FastAPI(title="Reminix Runtime", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "ok"}

    @app.get("/manifest")
    async def manifest() -> dict[str, Any]:
        """Runtime discovery endpoint."""

        endpoints: list[dict[str, Any]] = [
            {
                "kind": "agent",
                "path": f"/agents/{a.name}/invoke",
                "name": a.name,
                **a.metadata,
            }
            for a in agents
        ]

        if tools:
            endpoints.append({"kind": "mcp", "path": "/mcp"})

        return {
            "runtime": {
                "name": "reminix-runtime",
                "version": __version__,
                "language": "python",
            },
            "endpoints": endpoints,
        }

    @app.post("/agents/{agent_name}/invoke", response_model=None)
    async def invoke(
        agent_name: str, body: dict[str, Any]
    ) -> dict[str, Any] | StreamingResponse | JSONResponse:
        """Invoke an agent."""
        agent = agent_map.get(agent_name)
        if agent is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {"type": "NotFoundError", "message": f"Agent '{agent_name}' not found"}
                },
            )

        request = AgentRequest(
            input=body.get("input", {}),
            context=body.get("context"),
            stream=body.get("stream", False),
        )

        if request.stream:
            return StreamingResponse(
                _sse_generator(agent.invoke_stream(request)),
                media_type="text/event-stream",
            )

        try:
            return await agent.invoke(request)
        except NotImplementedError as e:
            return JSONResponse(
                status_code=501,
                content=_create_error_response(e, "NotImplementedError"),
            )
        except ValueError as e:
            return JSONResponse(
                status_code=400,
                content=_create_error_response(e, "ValidationError"),
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content=_create_error_response(e, type(e).__name__),
            )

    # Mount MCP Streamable HTTP endpoint
    from mcp.server.fastmcp.server import StreamableHTTPASGIApp
    from starlette.applications import Starlette
    from starlette.routing import Route

    mcp_asgi = StreamableHTTPASGIApp(mcp_session_manager)
    mcp_starlette = Starlette(routes=[Route("/mcp", endpoint=mcp_asgi)])
    app.mount("/", mcp_starlette)

    return app


def serve(
    *,
    agents: list[Agent] | None = None,
    tools: list[Tool] | None = None,
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
