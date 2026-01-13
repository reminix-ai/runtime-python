"""LangGraph adapter for Reminix Runtime."""

from typing import Any

from reminix_runtime import BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangGraph agents."""

    def __init__(self, agent: Any, name: str = "langgraph-agent") -> None:
        self._agent = agent
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request."""
        # TODO: Implement
        raise NotImplementedError()

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request."""
        # TODO: Implement
        raise NotImplementedError()


def wrap(agent: Any, name: str = "langgraph-agent") -> LangGraphAdapter:
    """Wrap a LangGraph agent for use with Reminix Runtime.

    Args:
        agent: A LangGraph compiled graph.
        name: Name for the agent.

    Returns:
        A LangGraphAdapter instance.
    """
    return LangGraphAdapter(agent, name=name)
