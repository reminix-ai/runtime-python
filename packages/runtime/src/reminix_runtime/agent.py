"""Agent classes for Reminix Runtime."""
# ruff: noqa: ARG002  # abstract methods have unused args in interface definitions

import inspect
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar, get_type_hints

from docstring_parser import parse as parse_docstring

from . import __version__
from .tool import _python_type_to_json_schema
from .types import ExecuteRequest, ExecuteResponse, Message

# Default parameters schema for agents
# Request: { "prompt": "..." }
DEFAULT_AGENT_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "prompt": {"type": "string", "description": "The prompt or task for the agent"},
    },
    "required": ["prompt"],
}

# ASGI type aliases
Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# Type aliases for handlers
ExecuteHandler = Callable[[ExecuteRequest], Awaitable[ExecuteResponse]]
ExecuteStreamHandler = Callable[[ExecuteRequest], AsyncIterator[str]]

F = TypeVar("F", bound=Callable[..., Any])


class AgentBase(ABC):
    """Abstract base class defining the agent interface.

    This is the core contract that all agents must fulfill.
    Use `Agent` for decorator-based registration or extend
    `AgentAdapter` for framework adapters.
    """

    @property
    def streaming(self) -> bool:
        """Whether this agent supports streaming requests."""
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
        return {
            "type": "agent",
            "parameters": DEFAULT_AGENT_PARAMETERS,
            "requestKeys": ["prompt"],
            "responseKeys": ["content"],
        }

    @abstractmethod
    async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        """Handle an execute request."""
        ...

    async def execute_stream(self, request: ExecuteRequest) -> AsyncIterator[str]:
        """Handle a streaming execute request."""
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

            @agent.on_execute
            async def handle(request):
                return ExecuteResponse(output="Hello!")

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
                                    "streaming": agent.streaming,
                                }
                            ],
                        }
                    )
                    return

                # POST /agents/{name}/execute
                execute_match = re.match(r"^/agents/([^/]+)/execute$", path)
                if method == "POST" and execute_match:
                    agent_name = execute_match.group(1)
                    if agent_name != agent.name:
                        await json_response({"error": f"Agent '{agent_name}' not found"}, 404)
                        return

                    body_bytes = await read_body()
                    body = json.loads(body_bytes)

                    # Get requestKeys from agent metadata
                    request_keys = agent.metadata.get("requestKeys", [])

                    # Extract declared keys from body into input
                    input_data: dict[str, Any] = {}
                    for key in request_keys:
                        if key in body:
                            input_data[key] = body[key]

                    request = ExecuteRequest(
                        input=input_data,
                        context=body.get("context"),
                        stream=body.get("stream", False),
                    )

                    if request.stream:
                        await sse_response(agent.execute_stream(request))
                        return

                    response = await agent.execute(request)
                    await json_response(response)
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

        @agent.on_execute
        async def handle_execute(request: ExecuteRequest) -> ExecuteResponse:
            return ExecuteResponse(output="Hello!")

        serve(agents=[agent], port=8080)
    """

    def __init__(self, name: str, *, metadata: dict[str, Any] | None = None):
        """Create a new agent.

        Args:
            name: The agent name (used in URLs like /agents/{name}/execute)
            metadata: Optional metadata for discovery
        """
        self._name = name
        self._metadata = metadata or {}

        # Handler storage
        self._execute_handler: ExecuteHandler | None = None
        self._execute_stream_handler: ExecuteStreamHandler | None = None

    @property
    def name(self) -> str:
        """Return the agent name."""
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return agent metadata for discovery."""
        return {
            "type": "agent",
            "parameters": DEFAULT_AGENT_PARAMETERS,
            "requestKeys": ["prompt"],
            "responseKeys": ["content"],
            **self._metadata,
        }

    @property
    def streaming(self) -> bool:
        """Whether execute supports streaming."""
        return self._execute_stream_handler is not None

    # Decorator methods for handler registration

    def on_execute(self, fn: ExecuteHandler) -> ExecuteHandler:
        """Register an execute handler.

        Example:
            @agent.on_execute
            async def handle(request: ExecuteRequest) -> ExecuteResponse:
                return ExecuteResponse(output="Hello!")
        """
        self._execute_handler = fn
        return fn

    def on_execute_stream(self, fn: ExecuteStreamHandler) -> ExecuteStreamHandler:
        """Register a streaming execute handler.

        Example:
            @agent.on_execute_stream
            async def handle(request: ExecuteRequest):
                yield '{"chunk": "Hello"}'
                yield '{"chunk": " world!"}'
        """
        self._execute_stream_handler = fn
        return fn

    # Implementation of abstract methods

    async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        """Handle an execute request."""
        if self._execute_handler is None:
            raise NotImplementedError(f"No execute handler registered for agent '{self._name}'")
        return await self._execute_handler(request)

    async def execute_stream(self, request: ExecuteRequest) -> AsyncIterator[str]:
        """Handle a streaming execute request."""
        if self._execute_stream_handler is None:
            raise NotImplementedError(
                f"No streaming execute handler registered for agent '{self._name}'"
            )
        async for chunk in self._execute_stream_handler(request):
            yield chunk


def _extract_parameters_from_function(
    func: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Extract JSON Schema parameters and output from function signature.

    Parses docstrings (Google, NumPy, or Sphinx style) to extract:
    - Parameter descriptions from Args section
    - Return description from Returns section

    Returns:
        A tuple of (parameters_schema, output_schema).
        parameters_schema is a dict with 'type', 'properties', and 'required' keys.
        output_schema is the JSON schema for the return type, or None if not specified.
    """
    # Parse docstring for parameter descriptions
    docstring = func.__doc__ or ""
    parsed_doc = parse_docstring(docstring)

    # Build parameter descriptions lookup from docstring Args section
    param_descriptions: dict[str, str] = {}
    for param in parsed_doc.params:
        if param.description:
            param_descriptions[param.arg_name] = param.description

    hints = get_type_hints(func)
    return_type = hints.pop("return", None)  # Extract return type hint
    output_schema = _python_type_to_json_schema(return_type) if return_type else None

    # Add return description to output schema if available
    if output_schema and parsed_doc.returns and parsed_doc.returns.description:
        output_schema["description"] = parsed_doc.returns.description

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

        # Add description from docstring if available
        if param_name in param_descriptions:
            schema["description"] = param_descriptions[param_name]

        properties[param_name] = schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            properties[param_name]["default"] = param.default

    return {"type": "object", "properties": properties, "required": required}, output_schema


def _wrap_output_schema_for_response_keys(
    output_schema: dict[str, Any] | None, response_keys: list[str]
) -> dict[str, Any] | None:
    """Wrap output schema to match the full response structure based on responseKeys.

    If responseKeys = ["output"], wraps the schema as { output: <schema> }
    If responseKeys = ["message"], wraps the schema as { message: <schema> }
    If responseKeys = ["message", "output"], wraps as { message: <schema>, output: <schema> }

    Args:
        output_schema: The schema for the return value (or None)
        response_keys: List of top-level response keys

    Returns:
        Wrapped schema describing the full response object, or None if output_schema is None
    """
    if output_schema is None or not response_keys:
        return None

    # If single response key, wrap the output schema
    if len(response_keys) == 1:
        return {
            "type": "object",
            "properties": {response_keys[0]: output_schema},
            "required": response_keys,
        }

    # Multiple response keys - need to split the output schema
    # For now, assume the output schema describes the first key's value
    # and other keys are optional/unknown
    properties: dict[str, Any] = {response_keys[0]: output_schema}
    required = [response_keys[0]]

    # For additional keys, we don't know their schema, so mark as optional
    # Users can override via metadata if they need full schema
    for key in response_keys[1:]:
        properties[key] = {"type": "object"}  # Placeholder - should be overridden

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def agent(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Agent | Callable[[Callable[..., Any]], Agent]:
    """Decorator to create an agent from a function.

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
        An Agent instance that handles execute requests.
    """

    def decorator(f: Callable[..., Any]) -> Agent:
        agent_name = name or f.__name__
        agent_description = description or (f.__doc__ or "").strip().split("\n\n")[0].strip()

        # Detect if streaming (async generator function)
        is_streaming = inspect.isasyncgenfunction(f)

        # Extract parameter and output schemas
        parameters, output = _extract_parameters_from_function(f)

        # Derive requestKeys from parameters properties
        request_keys = list(parameters.get("properties", {}).keys())

        # Default responseKeys (can be overridden via metadata)
        response_keys = ["content"]

        # Wrap output schema to match responseKeys structure
        wrapped_output = _wrap_output_schema_for_response_keys(output, response_keys)

        # Build metadata
        metadata: dict[str, Any] = {
            "description": agent_description,
            "parameters": parameters,
            "requestKeys": request_keys,
            "responseKeys": response_keys,
        }
        if wrapped_output is not None:
            metadata["output"] = wrapped_output

        # Create agent instance
        agent_instance = Agent(
            agent_name,
            metadata=metadata,
        )

        # Helper to get response keys from metadata (allows override)
        def get_response_keys() -> list[str]:
            keys = agent_instance.metadata.get("responseKeys", ["content"])
            return keys if keys else ["content"]

        if is_streaming:
            # Register streaming execute handler
            async def execute_stream_handler(request: ExecuteRequest) -> AsyncIterator[str]:
                async for chunk in f(**request.input):
                    # Convert to JSON string if not already a string
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        yield json.dumps(chunk)

            agent_instance.on_execute_stream(execute_stream_handler)

            # Also register non-streaming handler that collects chunks
            async def execute_handler(request: ExecuteRequest) -> ExecuteResponse:
                chunks: list[str] = []
                async for chunk in f(**request.input):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    else:
                        chunks.append(str(chunk))
                result = "".join(chunks)
                # If result is dict, use as-is; otherwise wrap in first responseKey
                if isinstance(result, dict):
                    return result
                response_keys = get_response_keys()
                return {response_keys[0]: result}

            agent_instance.on_execute(execute_handler)
        else:
            # Register regular execute handler
            async def execute_handler(request: ExecuteRequest) -> ExecuteResponse:
                if inspect.iscoroutinefunction(f):
                    result = await f(**request.input)
                else:
                    result = f(**request.input)
                # If result is dict, use as-is; otherwise wrap in first responseKey
                if isinstance(result, dict):
                    return result
                response_keys = get_response_keys()
                return {response_keys[0]: result}

            agent_instance.on_execute(execute_handler)

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

    The function receives messages and optionally context, returns a list of Message objects.
    Use async generator for streaming support.

    This is a convenience decorator that creates an agent with a standard chat
    interface (messages in, messages out).

    Request: { "messages": [...] }
    Response: { "messages": [{ "role": "assistant", "content": "..." }, ...] }

    Examples:
        @chat_agent
        async def assistant(messages: list[Message]) -> list[Message]:
            '''A helpful assistant.'''
            last_msg = messages[-1].content
            return [Message(role="assistant", content=f"You said: {last_msg}")]

        # With context
        @chat_agent
        async def contextual_bot(messages: list[Message], context: dict | None = None) -> list[Message]:
            '''Bot with context.'''
            user_id = context.get("user_id") if context else None
            return [Message(role="assistant", content=f"Hello user {user_id}!")]

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
        An Agent instance that handles execute requests with chat semantics.
    """

    def decorator(f: Callable[..., Any]) -> Agent:
        agent_name = name or f.__name__
        agent_description = description or (f.__doc__ or "").strip().split("\n\n")[0].strip()

        # Detect if streaming (async generator function)
        is_streaming = inspect.isasyncgenfunction(f)

        # Check if function accepts context parameter
        sig = inspect.signature(f)
        accepts_context = "context" in sig.parameters

        # Chat agents have default request/response keys (can be overridden via metadata)
        request_keys = ["messages"]
        response_keys = ["messages"]

        # Message item schema (shared between parameters and output)
        message_item_schema = {
            "type": "object",
            "properties": {
                "role": {
                    "type": "string",
                    "enum": ["system", "user", "assistant", "tool"],
                },
                "content": {
                    "type": ["string", "null"],
                },
                "name": {
                    "type": "string",
                },
                "tool_call_id": {
                    "type": "string",
                },
                "tool_calls": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "type": {"type": "string", "enum": ["function"]},
                            "function": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "arguments": {"type": "string"},
                                },
                                "required": ["name", "arguments"],
                            },
                        },
                        "required": ["id", "type", "function"],
                    },
                },
            },
            "required": ["role"],
        }

        # Define standard chat agent schemas
        parameters_schema = {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "items": message_item_schema,
                }
            },
            "required": ["messages"],
        }
        # Messages schema (array of messages, the value, not the full response)
        messages_schema = {
            "type": "array",
            "items": message_item_schema,
        }

        # Wrap messages schema to match responseKeys structure
        wrapped_output = _wrap_output_schema_for_response_keys(messages_schema, response_keys)

        # Build metadata
        metadata: dict[str, Any] = {
            "type": "chat_agent",
            "description": agent_description,
            "parameters": parameters_schema,
            "requestKeys": request_keys,
            "responseKeys": response_keys,
        }
        if wrapped_output is not None:
            metadata["output"] = wrapped_output

        # Create agent instance
        agent_instance = Agent(
            agent_name,
            metadata=metadata,
        )

        # Helper to get response keys from metadata (allows override)
        def get_response_keys() -> list[str]:
            keys = agent_instance.metadata.get("responseKeys", ["messages"])
            return keys if keys else ["messages"]

        if is_streaming:
            # Register streaming execute handler
            async def execute_stream_handler(request: ExecuteRequest) -> AsyncIterator[str]:
                # Extract messages from input
                raw_messages = request.input.get("messages", [])
                messages = [Message(**m) for m in raw_messages]
                kwargs: dict[str, Any] = {"messages": messages}
                if accepts_context:
                    kwargs["context"] = request.context
                async for chunk in f(**kwargs):
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        yield json.dumps(chunk)

            agent_instance.on_execute_stream(execute_stream_handler)

            # Also register non-streaming handler that collects chunks
            async def execute_handler(request: ExecuteRequest) -> ExecuteResponse:
                raw_messages = request.input.get("messages", [])
                messages = [Message(**m) for m in raw_messages]
                kwargs: dict[str, Any] = {"messages": messages}
                if accepts_context:
                    kwargs["context"] = request.context
                chunks: list[str] = []
                async for chunk in f(**kwargs):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    else:
                        chunks.append(str(chunk))
                result = [{"role": "assistant", "content": "".join(chunks)}]
                # For chat agents, always wrap in first responseKey (typically "messages")
                response_keys = get_response_keys()
                return {response_keys[0]: result}

            agent_instance.on_execute(execute_handler)
        else:
            # Register regular execute handler
            async def execute_handler(request: ExecuteRequest) -> ExecuteResponse:
                raw_messages = request.input.get("messages", [])
                messages = [Message(**m) for m in raw_messages]
                kwargs: dict[str, Any] = {"messages": messages}
                if accepts_context:
                    kwargs["context"] = request.context

                if inspect.iscoroutinefunction(f):
                    result = await f(**kwargs)
                else:
                    result = f(**kwargs)

                # Convert list of Message objects to list of dicts
                if isinstance(result, list):
                    messages_list = [
                        {"role": m.role, "content": m.content} if isinstance(m, Message) else m
                        for m in result
                    ]
                elif isinstance(result, Message):
                    # Single Message returned, wrap in list
                    messages_list = [{"role": result.role, "content": result.content}]
                else:
                    messages_list = result

                # Check if result is already a full response dict with all responseKeys
                response_keys = get_response_keys()
                if isinstance(messages_list, dict) and all(
                    key in messages_list for key in response_keys
                ):
                    return messages_list
                # Otherwise wrap in first responseKey (typically "messages" for chat agents)
                return {response_keys[0]: messages_list}

            agent_instance.on_execute(execute_handler)

        # Preserve function metadata
        agent_instance.__doc__ = f.__doc__
        agent_instance.__name__ = f.__name__  # type: ignore[attr-defined]
        agent_instance.__module__ = f.__module__

        return agent_instance

    # Handle both @chat_agent and @chat_agent(...) syntax
    if func is not None:
        return decorator(func)

    return decorator
