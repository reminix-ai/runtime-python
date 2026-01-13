"""Base agent and adapter interface."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..types import InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


class Agent(ABC):
    """Base class for all agents and adapters."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent name."""
        ...

    @abstractmethod
    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request."""
        ...

    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request."""
        ...

    async def invoke_stream(
        self, request: InvokeRequest
    ) -> AsyncIterator[str]:
        """Handle a streaming invoke request."""
        raise NotImplementedError("Streaming not implemented for this adapter")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[str]:
        """Handle a streaming chat request."""
        raise NotImplementedError("Streaming not implemented for this adapter")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]


# Alias for backwards compatibility
BaseAdapter = Agent
