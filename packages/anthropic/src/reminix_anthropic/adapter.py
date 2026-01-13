"""Anthropic adapter for Reminix Runtime."""

from typing import Any

from reminix_runtime import BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic agents."""

    def __init__(self, agent: Any, name: str = "anthropic-agent") -> None:
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


def wrap(agent: Any, name: str = "anthropic-agent") -> AnthropicAdapter:
    """Wrap an Anthropic agent for use with Reminix Runtime.

    Args:
        agent: An Anthropic agent.
        name: Name for the agent.

    Returns:
        An AnthropicAdapter instance.
    """
    return AnthropicAdapter(agent, name=name)
