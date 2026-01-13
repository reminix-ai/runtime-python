"""Base agent and adapter interface."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Callable, Awaitable, TypeVar

from ..types import InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

# Type aliases for handlers
InvokeHandler = Callable[[InvokeRequest], Awaitable[InvokeResponse]]
ChatHandler = Callable[[ChatRequest], Awaitable[ChatResponse]]
InvokeStreamHandler = Callable[[InvokeRequest], AsyncIterator[str]]
ChatStreamHandler = Callable[[ChatRequest], AsyncIterator[str]]

F = TypeVar("F", bound=Callable[..., Any])


class AgentBase(ABC):
    """Abstract base class defining the agent interface.

    This is the core contract that all agents must fulfill.
    Use `Agent` for decorator-based registration or extend
    `BaseAdapter` for framework adapters.
    """

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


class Agent(AgentBase):
    """Concrete agent with decorator-based handler registration.

    Use this class to create custom agents by registering handlers
    with decorators:

        agent = Agent("my-agent")

        @agent.on_invoke
        async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
            return InvokeResponse(output="Hello!")

        @agent.on_chat
        async def handle_chat(request: ChatRequest) -> ChatResponse:
            return ChatResponse(output="Hi!", messages=[...])

        serve([agent], port=8080)
    """

    def __init__(self, name: str, *, metadata: dict[str, Any] | None = None):
        """Create a new agent.

        Args:
            name: The agent name (used in URLs like /agents/{name}/invoke)
            metadata: Optional metadata for discovery
        """
        self._name = name
        self._metadata = metadata or {}

        # Handler storage
        self._invoke_handler: InvokeHandler | None = None
        self._chat_handler: ChatHandler | None = None
        self._invoke_stream_handler: InvokeStreamHandler | None = None
        self._chat_stream_handler: ChatStreamHandler | None = None

    @property
    def name(self) -> str:
        """Return the agent name."""
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return agent metadata for discovery."""
        return {"type": "agent", **self._metadata}

    @property
    def invoke_streaming(self) -> bool:
        """Whether invoke supports streaming."""
        return self._invoke_stream_handler is not None

    @property
    def chat_streaming(self) -> bool:
        """Whether chat supports streaming."""
        return self._chat_stream_handler is not None

    # Decorator methods for handler registration

    def on_invoke(self, fn: InvokeHandler) -> InvokeHandler:
        """Register an invoke handler.

        Example:
            @agent.on_invoke
            async def handle(request: InvokeRequest) -> InvokeResponse:
                return InvokeResponse(output="Hello!")
        """
        self._invoke_handler = fn
        return fn

    def on_chat(self, fn: ChatHandler) -> ChatHandler:
        """Register a chat handler.

        Example:
            @agent.on_chat
            async def handle(request: ChatRequest) -> ChatResponse:
                return ChatResponse(output="Hi!", messages=[...])
        """
        self._chat_handler = fn
        return fn

    def on_invoke_stream(self, fn: InvokeStreamHandler) -> InvokeStreamHandler:
        """Register a streaming invoke handler.

        Example:
            @agent.on_invoke_stream
            async def handle(request: InvokeRequest):
                yield '{"chunk": "Hello"}'
                yield '{"chunk": " world!"}'
        """
        self._invoke_stream_handler = fn
        return fn

    def on_chat_stream(self, fn: ChatStreamHandler) -> ChatStreamHandler:
        """Register a streaming chat handler.

        Example:
            @agent.on_chat_stream
            async def handle(request: ChatRequest):
                yield '{"chunk": "Hi"}'
        """
        self._chat_stream_handler = fn
        return fn

    # Implementation of abstract methods

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request."""
        if self._invoke_handler is None:
            raise NotImplementedError(f"No invoke handler registered for agent '{self._name}'")
        return await self._invoke_handler(request)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request."""
        if self._chat_handler is None:
            raise NotImplementedError(f"No chat handler registered for agent '{self._name}'")
        return await self._chat_handler(request)

    async def invoke_stream(self, request: InvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request."""
        if self._invoke_stream_handler is None:
            raise NotImplementedError(f"No streaming invoke handler registered for agent '{self._name}'")
        async for chunk in self._invoke_stream_handler(request):
            yield chunk

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request."""
        if self._chat_stream_handler is None:
            raise NotImplementedError(f"No streaming chat handler registered for agent '{self._name}'")
        async for chunk in self._chat_stream_handler(request):
            yield chunk


class BaseAdapter(AgentBase):
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
