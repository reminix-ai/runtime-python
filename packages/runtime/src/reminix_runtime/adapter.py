"""Base adapter class for framework integrations."""
# ruff: noqa: ARG002  # abstract methods have unused args in interface definitions

from collections.abc import AsyncIterator
from typing import Any

from .agent import AgentBase
from .types import ChatRequest, InvokeRequest


class AdapterBase(AgentBase):
    """Base class for framework adapters.

    Extend this class when wrapping an existing AI framework
    (e.g., LangChain, OpenAI, Anthropic).
    """

    # Subclasses should override this with the adapter name
    adapter_name: str = "unknown"

    @property
    def invoke_streaming(self) -> bool:
        """Whether this adapter supports streaming invoke requests."""
        return True

    @property
    def chat_streaming(self) -> bool:
        """Whether this adapter supports streaming chat requests."""
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata for discovery."""
        return {"type": "adapter", "adapter": self.adapter_name}

    async def invoke_stream(self, request: InvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request."""
        raise NotImplementedError("Streaming not implemented for this adapter")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request."""
        raise NotImplementedError("Streaming not implemented for this adapter")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]
