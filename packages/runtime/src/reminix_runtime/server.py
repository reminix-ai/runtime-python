"""Reminix Runtime Server."""

import json
import os
import traceback
from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .agent import AgentBase
from .tool import ToolBase
from .types import (
    InvokeRequest,
    InvokeResponse,
    InvokeResponseDict,
)

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


async def _sse_generator(stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Convert an async string iterator to SSE format."""
    try:
        async for chunk in stream:
            data = json.dumps({"delta": chunk})
            yield f"data: {data}\n\n".encode()
        yield f"data: {json.dumps({'done': True})}\n\n".encode()
    except NotImplementedError as e:
        error_data = _create_error_response(e, "NotImplementedError")
        yield f"data: {json.dumps(error_data)}\n\n".encode()
    except Exception as e:
        error_data = _create_error_response(e, type(e).__name__)
        yield f"data: {json.dumps(error_data)}\n\n".encode()


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
    async def invoke(
        agent_name: str, body: dict[str, Any]
    ) -> InvokeResponseDict | StreamingResponse | JSONResponse:
        """Invoke an agent."""
        agent = agent_map.get(agent_name)
        if agent is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {"type": "NotFoundError", "message": f"Agent '{agent_name}' not found"}
                },
            )

        request = InvokeRequest(
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

    @app.post("/tools/{tool_name}/call", response_model=None)
    async def call_tool(tool_name: str, body: dict[str, Any]) -> InvokeResponse | JSONResponse:
        """Call a tool."""
        tool = tool_map.get(tool_name)
        if tool is None:
            return JSONResponse(
                status_code=404,
                content={
                    "error": {"type": "NotFoundError", "message": f"Tool '{tool_name}' not found"}
                },
            )

        request = InvokeRequest(
            input=body.get("input", {}),
            context=body.get("context"),
        )

        try:
            return await tool.call(request)
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
