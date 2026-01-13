"""LangChain adapter for Reminix Runtime."""

from typing import Any

from reminix_runtime import BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain agents."""

    def __init__(self, agent: Any, name: str = "langchain-agent") -> None:
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


def wrap(agent: Any, name: str = "langchain-agent") -> LangChainAdapter:
    """Wrap a LangChain agent for use with Reminix Runtime.

    Args:
        agent: A LangChain agent or runnable.
        name: Name for the agent.

    Returns:
        A LangChainAdapter instance.
    """
    return LangChainAdapter(agent, name=name)
