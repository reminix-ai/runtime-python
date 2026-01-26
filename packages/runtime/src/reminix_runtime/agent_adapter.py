"""Base agent adapter class for framework integrations."""
# ruff: noqa: ARG002  # abstract methods have unused args in interface definitions

from collections.abc import AsyncIterator
from typing import Any

from .agent import AgentBase
from .types import ExecuteRequest


class AgentAdapter(AgentBase):
    """Base class for framework agent adapters.

    Extend this class when wrapping an existing AI framework's agent
    (e.g., LangChain, OpenAI, Anthropic).
    """

    # Subclasses should override this with the adapter name
    adapter_name: str = "unknown"

    @property
    def streaming(self) -> bool:
        """Whether this adapter supports streaming execute requests."""
        return True

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata for discovery.

        Adapters accept both 'messages' (chat-style) and 'prompt' (simple) inputs.
        """
        return {
            "type": "adapter",
            "adapter": self.adapter_name,
            "requestKeys": ["messages", "prompt"],
            "responseKeys": ["output"],
        }

    async def execute_stream(self, request: ExecuteRequest) -> AsyncIterator[str]:
        """Handle a streaming execute request."""
        raise NotImplementedError("Streaming not implemented for this adapter")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]
