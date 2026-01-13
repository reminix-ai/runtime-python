"""LlamaIndex adapter for Reminix Runtime."""

from typing import Any

from reminix_runtime import BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


class LlamaIndexAdapter(BaseAdapter):
    """Adapter for LlamaIndex agents."""

    def __init__(self, agent: Any, name: str = "llamaindex-agent") -> None:
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


def wrap(agent: Any, name: str = "llamaindex-agent") -> LlamaIndexAdapter:
    """Wrap a LlamaIndex agent for use with Reminix Runtime.

    Args:
        agent: A LlamaIndex agent.
        name: Name for the agent.

    Returns:
        A LlamaIndexAdapter instance.
    """
    return LlamaIndexAdapter(agent, name=name)
