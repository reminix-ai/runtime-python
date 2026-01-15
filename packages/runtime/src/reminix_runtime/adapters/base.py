"""Base agent and adapter interface."""
# ruff: noqa: ARG002  # abstract methods have unused args in interface definitions

import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

from .. import __version__
from ..types import ChatRequest, ChatResponse, InvokeRequest, InvokeResponse, Message

# ASGI type aliases
Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

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

    @property
    def invoke_streaming(self) -> bool:
        """Whether this agent supports streaming invoke requests."""
        return False

    @property
    def chat_streaming(self) -> bool:
        """Whether this agent supports streaming chat requests."""
        return False

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

    async def invoke_stream(self, request: InvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request."""
        raise NotImplementedError("Streaming not implemented for this agent")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request."""
        raise NotImplementedError("Streaming not implemented for this agent")
        # Unreachable, but required to make this an async generator
        yield  # type: ignore[misc]

    def to_asgi(self) -> ASGIApp:
        """Create an ASGI application for this agent.

        Works with any ASGI server (uvicorn, hypercorn, daphne) or serverless
        platforms that support ASGI (AWS Lambda with Mangum, etc.).

        Example:
            ```python
            from mangum import Mangum

            agent = Agent("my-agent")

            @agent.on_invoke
            async def handle(request):
                return {"output": "Hello!"}

            # AWS Lambda handler
            handler = Mangum(agent.to_asgi())
            ```
        """
        agent = self

        async def asgi_app(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] != "http":
                return

            path = scope["path"]
            method = scope["method"]

            # Helper to send JSON response
            async def json_response(data: Any, status: int = 200) -> None:
                body = json.dumps(data).encode("utf-8")
                await send(
                    {
                        "type": "http.response.start",
                        "status": status,
                        "headers": [
                            [b"content-type", b"application/json"],
                            [b"access-control-allow-origin", b"*"],
                            [b"access-control-allow-methods", b"GET, POST, OPTIONS"],
                            [b"access-control-allow-headers", b"content-type"],
                        ],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": body,
                    }
                )

            # Helper to send SSE stream
            async def sse_response(stream: AsyncIterator[str]) -> None:
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            [b"content-type", b"text/event-stream"],
                            [b"cache-control", b"no-cache"],
                            [b"connection", b"keep-alive"],
                            [b"access-control-allow-origin", b"*"],
                        ],
                    }
                )
                try:
                    async for chunk in stream:
                        await send(
                            {
                                "type": "http.response.body",
                                "body": f"data: {chunk}\n\n".encode(),
                                "more_body": True,
                            }
                        )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": b"data: [DONE]\n\n",
                            "more_body": False,
                        }
                    )
                except NotImplementedError as e:
                    await send(
                        {
                            "type": "http.response.body",
                            "body": f'data: {{"error": "{str(e)}"}}\n\n'.encode(),
                            "more_body": False,
                        }
                    )

            # Helper to read request body
            async def read_body() -> bytes:
                body = b""
                while True:
                    message = await receive()
                    body += message.get("body", b"")
                    if not message.get("more_body", False):
                        break
                return body

            # Handle CORS preflight
            if method == "OPTIONS":
                await send(
                    {
                        "type": "http.response.start",
                        "status": 204,
                        "headers": [
                            [b"access-control-allow-origin", b"*"],
                            [b"access-control-allow-methods", b"GET, POST, OPTIONS"],
                            [b"access-control-allow-headers", b"content-type"],
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": b""})
                return

            try:
                # GET /health
                if method == "GET" and path == "/health":
                    await json_response({"status": "ok"})
                    return

                # GET /info
                if method == "GET" and path == "/info":
                    await json_response(
                        {
                            "runtime": {
                                "name": "reminix-runtime",
                                "version": __version__,
                                "language": "python",
                                "framework": "asgi",
                            },
                            "agents": [
                                {
                                    "name": agent.name,
                                    **agent.metadata,
                                    "invoke": {"streaming": agent.invoke_streaming},
                                    "chat": {"streaming": agent.chat_streaming},
                                }
                            ],
                        }
                    )
                    return

                # POST /agents/{name}/invoke
                invoke_match = re.match(r"^/agents/([^/]+)/invoke$", path)
                if method == "POST" and invoke_match:
                    agent_name = invoke_match.group(1)
                    if agent_name != agent.name:
                        await json_response({"error": f"Agent '{agent_name}' not found"}, 404)
                        return

                    body_bytes = await read_body()
                    body = json.loads(body_bytes)

                    if not body.get("input"):
                        await json_response(
                            {"error": "input is required and must not be empty"}, 400
                        )
                        return

                    request = InvokeRequest(
                        input=body["input"],
                        context=body.get("context"),
                        stream=body.get("stream", False),
                    )

                    if request.stream:
                        await sse_response(agent.invoke_stream(request))
                        return

                    response = await agent.invoke(request)
                    await json_response({"output": response.output})
                    return

                # POST /agents/{name}/chat
                chat_match = re.match(r"^/agents/([^/]+)/chat$", path)
                if method == "POST" and chat_match:
                    agent_name = chat_match.group(1)
                    if agent_name != agent.name:
                        await json_response({"error": f"Agent '{agent_name}' not found"}, 404)
                        return

                    body_bytes = await read_body()
                    body = json.loads(body_bytes)

                    if not body.get("messages"):
                        await json_response(
                            {"error": "messages is required and must not be empty"}, 400
                        )
                        return

                    messages = [Message(**m) for m in body["messages"]]
                    request = ChatRequest(
                        messages=messages,
                        context=body.get("context"),
                        stream=body.get("stream", False),
                    )

                    if request.stream:
                        await sse_response(agent.chat_stream(request))
                        return

                    response = await agent.chat(request)
                    await json_response(
                        {
                            "output": response.output,
                            "messages": response.messages,  # Already list[dict]
                        }
                    )
                    return

                # Not found
                await json_response({"error": "Not found"}, 404)

            except Exception as e:
                await json_response({"error": str(e)}, 500)

        return asgi_app


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
            raise NotImplementedError(
                f"No streaming invoke handler registered for agent '{self._name}'"
            )
        async for chunk in self._invoke_stream_handler(request):
            yield chunk

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request."""
        if self._chat_stream_handler is None:
            raise NotImplementedError(
                f"No streaming chat handler registered for agent '{self._name}'"
            )
        async for chunk in self._chat_stream_handler(request):
            yield chunk


class BaseAdapter(AgentBase):
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
