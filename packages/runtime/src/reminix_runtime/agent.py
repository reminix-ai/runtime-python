"""Agent classes for Reminix Runtime."""
# ruff: noqa: ARG002  # abstract methods have unused args in interface definitions

import inspect
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar, get_type_hints

from . import __version__
from .types import ChatRequest, ChatResponse, InvokeRequest, InvokeResponse, Message

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

        serve(agents=[agent], port=8080)
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


# Type mapping from Python types to JSON Schema types (shared with tool.py)
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(python_type: type) -> dict[str, Any]:
    """Convert a Python type hint to a JSON Schema type."""
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle basic types
    if python_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[python_type]}

    # Handle Optional types (Union with None)
    origin = getattr(python_type, "__origin__", None)
    if origin is not None:
        args = getattr(python_type, "__args__", ())

        # Handle Union types (including Optional)
        if origin is type(None) or str(origin) == "typing.Union":
            non_none_args = [a for a in args if a is not type(None)]
            if len(non_none_args) == 1:
                return _python_type_to_json_schema(non_none_args[0])

        # Handle list[T]
        if origin is list:
            if args:
                return {"type": "array", "items": _python_type_to_json_schema(args[0])}
            return {"type": "array"}

        # Handle dict[K, V]
        if origin is dict:
            return {"type": "object"}

    # Default to string for unknown types
    return {"type": "string"}


def _extract_parameters_from_function(
    func: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Extract JSON Schema parameters and output from function signature.

    Returns:
        A tuple of (parameters_schema, output_schema).
        parameters_schema is a dict with 'type', 'properties', and 'required' keys.
        output_schema is the JSON schema for the return type, or None if not specified.
    """
    hints = get_type_hints(func)
    return_type = hints.pop("return", None)  # Extract return type hint
    output_schema = _python_type_to_json_schema(return_type) if return_type else None

    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip special parameters
        if param_name in ("self", "cls", "messages", "context"):
            continue

        # Get type hint
        param_type = hints.get(param_name, str)
        schema = _python_type_to_json_schema(param_type)

        properties[param_name] = schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["default"] = param.default

    return {"type": "object", "properties": properties, "required": required}, output_schema


def agent(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Agent | Callable[[Callable[..., Any]], Agent]:
    """Decorator to create an invoke agent from a function.

    The function parameters become the agent's input schema (like @tool).
    Supports both sync and async functions, and async generators for streaming.

    Examples:
        @agent
        async def calculator(a: float, b: float) -> float:
            '''Add two numbers.'''
            return a + b

        @agent(name="custom-name")
        async def echo(message: str) -> str:
            '''Echo the message.'''
            return f"Echo: {message}"

        # Streaming agent (yields chunks, collected for non-streaming requests)
        @agent
        async def streamer(text: str):
            '''Stream text word by word.'''
            for word in text.split():
                yield word + " "

    Args:
        func: The function to wrap (when used without parentheses).
        name: Optional name override. Defaults to function name.
        description: Optional description override. Defaults to docstring.

    Returns:
        An Agent instance that handles invoke requests.
    """

    def decorator(f: Callable[..., Any]) -> Agent:
        agent_name = name or f.__name__
        agent_description = description or (f.__doc__ or "").strip().split("\n\n")[0].strip()

        # Detect if streaming (async generator function)
        is_streaming = inspect.isasyncgenfunction(f)

        # Extract parameter and output schemas
        parameters, output = _extract_parameters_from_function(f)

        # Build metadata
        metadata: dict[str, Any] = {
            "description": agent_description,
            "parameters": parameters,
        }
        if output is not None:
            metadata["output"] = output

        # Create agent instance
        agent_instance = Agent(
            agent_name,
            metadata=metadata,
        )

        if is_streaming:
            # Register streaming invoke handler
            async def invoke_stream_handler(request: InvokeRequest) -> AsyncIterator[str]:
                async for chunk in f(**request.input):
                    # Convert to JSON string if not already a string
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        yield json.dumps(chunk)

            agent_instance.on_invoke_stream(invoke_stream_handler)

            # Also register non-streaming handler that collects chunks
            async def invoke_handler(request: InvokeRequest) -> InvokeResponse:
                chunks: list[str] = []
                async for chunk in f(**request.input):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    else:
                        chunks.append(str(chunk))
                return InvokeResponse(output="".join(chunks))

            agent_instance.on_invoke(invoke_handler)
        else:
            # Register regular invoke handler
            async def invoke_handler(request: InvokeRequest) -> InvokeResponse:
                if inspect.iscoroutinefunction(f):
                    result = await f(**request.input)
                else:
                    result = f(**request.input)
                return InvokeResponse(output=result)

            agent_instance.on_invoke(invoke_handler)

        # Preserve function metadata
        agent_instance.__doc__ = f.__doc__
        agent_instance.__name__ = f.__name__  # type: ignore[attr-defined]
        agent_instance.__module__ = f.__module__

        return agent_instance

    # Handle both @agent and @agent(...) syntax
    if func is not None:
        return decorator(func)

    return decorator


def chat_agent(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Agent | Callable[[Callable[..., Any]], Agent]:
    """Decorator to create a chat agent from a function.

    The function receives messages and optionally context, returns a response string.
    Use async generator for streaming support.

    Examples:
        @chat_agent
        async def assistant(messages: list[Message]) -> str:
            '''A helpful assistant.'''
            last_msg = messages[-1].content
            return f"You said: {last_msg}"

        # With context
        @chat_agent
        async def contextual_bot(messages: list[Message], context: dict | None = None) -> str:
            '''Bot with context.'''
            user_id = context.get("user_id") if context else None
            return f"Hello user {user_id}!"

        # Streaming (yields chunks, collected for non-streaming requests)
        @chat_agent
        async def streaming_assistant(messages: list[Message]):
            '''Streams responses token by token.'''
            for token in ["Hello", " ", "world", "!"]:
                yield token

    Args:
        func: The function to wrap (when used without parentheses).
        name: Optional name override. Defaults to function name.
        description: Optional description override. Defaults to docstring.

    Returns:
        An Agent instance that handles chat requests.
    """

    def decorator(f: Callable[..., Any]) -> Agent:
        agent_name = name or f.__name__
        agent_description = description or (f.__doc__ or "").strip().split("\n\n")[0].strip()

        # Detect if streaming (async generator function)
        is_streaming = inspect.isasyncgenfunction(f)

        # Check if function accepts context parameter
        sig = inspect.signature(f)
        accepts_context = "context" in sig.parameters

        # Define standard chat agent schemas
        parameters_schema = {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                }
            },
            "required": ["messages"],
        }
        output_schema = {"type": "string"}

        # Create agent instance
        agent_instance = Agent(
            agent_name,
            metadata={
                "description": agent_description,
                "parameters": parameters_schema,
                "output": output_schema,
            },
        )

        if is_streaming:
            # Register streaming chat handler
            async def chat_stream_handler(request: ChatRequest) -> AsyncIterator[str]:
                kwargs: dict[str, Any] = {"messages": request.messages}
                if accepts_context:
                    kwargs["context"] = request.context
                async for chunk in f(**kwargs):
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        yield json.dumps(chunk)

            agent_instance.on_chat_stream(chat_stream_handler)

            # Also register non-streaming handler that collects chunks
            async def chat_handler(request: ChatRequest) -> ChatResponse:
                kwargs: dict[str, Any] = {"messages": request.messages}
                if accepts_context:
                    kwargs["context"] = request.context
                chunks: list[str] = []
                async for chunk in f(**kwargs):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    else:
                        chunks.append(str(chunk))
                output = "".join(chunks)
                return ChatResponse(
                    output=output,
                    messages=[
                        *[{"role": m.role, "content": m.content} for m in request.messages],
                        {"role": "assistant", "content": output},
                    ],
                )

            agent_instance.on_chat(chat_handler)
        else:
            # Register regular chat handler
            async def chat_handler(request: ChatRequest) -> ChatResponse:
                kwargs: dict[str, Any] = {"messages": request.messages}
                if accepts_context:
                    kwargs["context"] = request.context

                if inspect.iscoroutinefunction(f):
                    output = await f(**kwargs)
                else:
                    output = f(**kwargs)

                return ChatResponse(
                    output=output,
                    messages=[
                        *[{"role": m.role, "content": m.content} for m in request.messages],
                        {"role": "assistant", "content": output},
                    ],
                )

            agent_instance.on_chat(chat_handler)

        # Preserve function metadata
        agent_instance.__doc__ = f.__doc__
        agent_instance.__name__ = f.__name__  # type: ignore[attr-defined]
        agent_instance.__module__ = f.__module__

        return agent_instance

    # Handle both @chat_agent and @chat_agent(...) syntax
    if func is not None:
        return decorator(func)

    return decorator
