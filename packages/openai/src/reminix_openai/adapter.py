"""OpenAI adapter for Reminix Runtime."""

from typing import Any

from reminix_runtime import BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI Agents."""

    def __init__(self, agent: Any, name: str = "openai-agent") -> None:
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


def wrap(agent: Any, name: str = "openai-agent") -> OpenAIAdapter:
    """Wrap an OpenAI agent for use with Reminix Runtime.

    Args:
        agent: An OpenAI agent.
        name: Name for the agent.

    Returns:
        An OpenAIAdapter instance.
    """
    return OpenAIAdapter(agent, name=name)
