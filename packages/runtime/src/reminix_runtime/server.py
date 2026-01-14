"""Reminix Runtime Server."""

from collections.abc import AsyncIterator
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import __version__
from .adapters.base import AgentBase
from .types import ChatRequest, ChatResponse, InvokeRequest, InvokeResponse


async def _sse_generator(stream: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Convert an async string iterator to SSE format."""
    try:
        async for chunk in stream:
            yield f"data: {chunk}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except NotImplementedError as e:
        yield f'data: {{"error": "{str(e)}"}}\n\n'.encode()


def create_app(agents: list[AgentBase]) -> FastAPI:
    """Create a FastAPI application with agent endpoints.

    Args:
        agents: List of agents.

    Returns:
        A FastAPI application instance.

    Raises:
        ValueError: If no agents are provided.
    """
    if not agents:
        raise ValueError("At least one agent is required")

    # Build a lookup dict for agents by name
    agent_map: dict[str, AgentBase] = {agent.name: agent for agent in agents}

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

    return app


def serve(agents: list[AgentBase], port: int = 8080, host: str = "0.0.0.0") -> None:
    """Serve agents via REST API.

    Args:
        agents: List of agents.
        port: Port to serve on.
        host: Host to bind to.
    """
    import uvicorn

    app = create_app(agents)
    uvicorn.run(app, host=host, port=port)
