"""Reminix Runtime Server."""

from typing import Any

from fastapi import FastAPI, HTTPException

from . import __version__
from .adapters.base import Agent
from .types import InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


def create_app(agents: list[Agent]) -> FastAPI:
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
    agent_map: dict[str, Agent] = {agent.name: agent for agent in agents}

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
                    "endpoints": {
                        "invoke": f"/agents/{agent.name}/invoke",
                        "chat": f"/agents/{agent.name}/chat",
                    },
                }
                for agent in agents
            ],
        }

    @app.post("/agents/{agent_name}/invoke")
    async def invoke(agent_name: str, request: InvokeRequest) -> InvokeResponse:
        """Invoke an agent."""
        agent = agent_map.get(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

        return await agent.invoke(request)

    @app.post("/agents/{agent_name}/chat")
    async def chat(agent_name: str, request: ChatRequest) -> ChatResponse:
        """Chat with an agent."""
        agent = agent_map.get(agent_name)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")

        return await agent.chat(request)

    return app


def serve(agents: list[Agent], port: int = 8080, host: str = "0.0.0.0") -> None:
    """Serve agents via REST API.

    Args:
        agents: List of agents.
        port: Port to serve on.
        host: Host to bind to.
    """
    import uvicorn

    app = create_app(agents)
    uvicorn.run(app, host=host, port=port)
