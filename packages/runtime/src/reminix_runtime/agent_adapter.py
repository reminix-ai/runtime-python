"""Base agent adapter class for framework integrations."""
# ruff: noqa: ARG002  # abstract methods have unused args in interface definitions

from collections.abc import AsyncIterator
from typing import Any

from .agent import AgentBase
from .types import AgentInvokeRequest

# Adapter input schema - accepts both messages and prompt
ADAPTER_INPUT: dict[str, Any] = {
    "type": "object",
    "properties": {
        "messages": {
            "type": "array",
            "description": "Chat-style messages input",
        },
        "prompt": {
            "type": "string",
            "description": "Simple prompt input",
        },
    },
}


class AgentAdapter(AgentBase):
    """Base class for framework agent adapters.

    Extend this class when wrapping an existing AI framework's agent
    (e.g., LangChain, OpenAI, Anthropic).
    """

    # Subclasses should override this with the adapter name
    adapter_name: str = "unknown"

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata for discovery.

        Adapters accept both 'messages' (chat-style) and 'prompt' (simple) inputs.
        """
        return {
            "description": f"{self.adapter_name} adapter",
            "capabilities": {
                "streaming": True,
            },
            "input": ADAPTER_INPUT,
            "output": {"type": "string"},
            "adapter": self.adapter_name,
        }

    async def invoke_stream(self, request: AgentInvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request."""
        raise NotImplementedError("Streaming not implemented for this adapter")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]
