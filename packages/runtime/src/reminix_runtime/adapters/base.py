"""Base agent and adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator

from ..types import InvokeRequest, InvokeResponse, ChatRequest, ChatResponse


class Agent(ABC):
    """Base class for all agents."""

    # Override these to indicate streaming support
    invoke_streaming: bool = False
    chat_streaming: bool = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the agent name."""
        ...

    @property
    def metadata(self) -> dict[str, Any]:
        """Return agent metadata for discovery.
        
        Override this to provide custom metadata.
        """
        return {"type": "agent"}

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
        raise NotImplementedError("Streaming not implemented for this agent")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[str]:
        """Handle a streaming chat request."""
        raise NotImplementedError("Streaming not implemented for this agent")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]


class BaseAdapter(Agent):
    """Base class for framework adapters.
    
    Extend this class when wrapping an existing AI framework
    (e.g., LangChain, OpenAI, Anthropic).
    """

    # Subclasses should override this with the adapter name
    adapter_name: str = "unknown"

    # All built-in adapters support streaming
    invoke_streaming: bool = True
    chat_streaming: bool = True

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata for discovery."""
        return {"type": "adapter", "adapter": self.adapter_name}

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
