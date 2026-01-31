"""Agent classes for Reminix Runtime."""
# ruff: noqa: ARG002  # abstract methods have unused args in interface definitions

import inspect
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, Literal, TypeVar, get_type_hints

from docstring_parser import parse as parse_docstring

from . import __version__
from .tool import _python_type_to_json_schema
from .types import AgentInvokeRequest, AgentInvokeResponseDict

# Named agent templates with predefined input/output schemas.
# The default template is 'prompt'; use it when no template or custom schema is provided.
AgentTemplate = Literal["prompt", "chat", "task", "rag", "thread"]

DEFAULT_AGENT_TEMPLATE: AgentTemplate = "prompt"

# JSON schema for a single tool call (OpenAI-style)
TOOL_CALL_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Tool call id"},
        "type": {"type": "string", "enum": ["function"], "description": "Tool call type"},
        "function": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function/tool name"},
                "arguments": {"type": "string", "description": "JSON string of arguments"},
            },
            "required": ["name", "arguments"],
        },
    },
    "required": ["id", "type", "function"],
}

# JSON schema for a message item (OpenAI-style; supports tool_calls and tool results)
MESSAGE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "description": "Message role (user, assistant, system, tool)"},
        "content": {"type": "string", "description": "Message content", "nullable": True},
        "tool_calls": {
            "type": "array",
            "description": "Tool calls requested by the model (assistant messages)",
            "items": TOOL_CALL_ITEM_SCHEMA,
        },
        "tool_call_id": {
            "type": "string",
            "description": "Id of the tool call this message is a result for (tool messages)",
        },
        "name": {"type": "string", "description": "Tool name (tool messages)"},
    },
}

AGENT_TEMPLATES: dict[AgentTemplate, dict[str, Any]] = {
    "prompt": {
        "input": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt or task for the agent"},
            },
            "required": ["prompt"],
        },
        "output": {"type": "string"},
    },
    "chat": {
        "input": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Chat messages (OpenAI-style)",
                    "items": MESSAGE_ITEM_SCHEMA,
                },
            },
            "required": ["messages"],
        },
        "output": {"type": "string"},
    },
    "task": {
        "input": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Task name or description"},
            },
            "required": ["task"],
            "additionalProperties": True,
        },
        "output": {
            "description": "Structured JSON result (object, array, string, number, boolean, or null)",
            "type": "object",
            "additionalProperties": True,
        },
    },
    "rag": {
        "input": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to answer from documents"},
                "messages": {
                    "type": "array",
                    "description": "Optional prior conversation (chat-style RAG)",
                    "items": MESSAGE_ITEM_SCHEMA,
                },
                "collectionIds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional knowledge collection IDs to scope the search",
                },
            },
            "required": ["query"],
        },
        "output": {"type": "string"},
    },
    "thread": {
        "input": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Chat messages with tool_calls and tool results (OpenAI-style)",
                    "items": MESSAGE_ITEM_SCHEMA,
                },
            },
            "required": ["messages"],
        },
        "output": {
            "type": "array",
            "description": "Updated message thread (OpenAI-style, may include assistant message and tool_calls)",
            "items": MESSAGE_ITEM_SCHEMA,
        },
    },
}

# Default input/output schemas (same as prompt template). Used by AgentBase and custom agents.
DEFAULT_AGENT_INPUT: dict[str, Any] = AGENT_TEMPLATES[DEFAULT_AGENT_TEMPLATE]["input"]
DEFAULT_AGENT_OUTPUT: dict[str, Any] = AGENT_TEMPLATES[DEFAULT_AGENT_TEMPLATE]["output"]

# ASGI type aliases
Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict[str, Any]]]
Send = Callable[[dict[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# Type aliases for handlers
InvokeHandler = Callable[[AgentInvokeRequest], Awaitable[AgentInvokeResponseDict]]
InvokeStreamHandler = Callable[[AgentInvokeRequest], AsyncIterator[str]]

F = TypeVar("F", bound=Callable[..., Any])


class AgentBase(ABC):
    """Abstract base class defining the agent interface.

    This is the core contract that all agents must fulfill.
    Use `Agent` for decorator-based registration or extend
    `AgentAdapter` for framework adapters.
    """

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
            "capabilities": {"streaming": False},
            "input": DEFAULT_AGENT_INPUT,
            "output": DEFAULT_AGENT_OUTPUT,
        }

    @abstractmethod
    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponseDict:
        """Handle an invoke request."""
        ...

    async def invoke_stream(self, request: AgentInvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request."""
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

            @agent.handler
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
                        data = json.dumps({"delta": chunk})
                        await send(
                            {
                                "type": "http.response.body",
                                "body": f"data: {data}\n\n".encode(),
                                "more_body": True,
                            }
                        )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": f"data: {json.dumps({'done': True})}\n\n".encode(),
                            "more_body": False,
                        }
                    )
                except NotImplementedError as e:
                    error_data = {"error": {"type": "NotImplementedError", "message": str(e)}}
                    await send(
                        {
                            "type": "http.response.body",
                            "body": f"data: {json.dumps(error_data)}\n\n".encode(),
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
                        await json_response(
                            {
                                "error": {
                                    "type": "NotFoundError",
                                    "message": f"Agent '{agent_name}' not found",
                                }
                            },
                            404,
                        )
                        return

                    body_bytes = await read_body()
                    body = json.loads(body_bytes)

                    request = AgentInvokeRequest(
                        input=body.get("input", {}),
                        context=body.get("context"),
                        stream=body.get("stream", False),
                    )

                    if request.stream:
                        await sse_response(agent.invoke_stream(request))
                        return

                    response = await agent.invoke(request)
                    await json_response(response)
                    return

                # Not found
                await json_response(
                    {"error": {"type": "NotFoundError", "message": "Not found"}},
                    404,
                )

            except Exception as e:
                await json_response(
                    {"error": {"type": type(e).__name__, "message": str(e)}},
                    500,
                )

        return asgi_app


class Agent(AgentBase):
    """Concrete agent with decorator-based handler registration.

    Use this class to create custom agents by registering handlers
    with decorators:

        agent = Agent("my-agent")

        @agent.handler
        async def handle_invoke(request: AgentInvokeRequest) -> AgentInvokeResponseDict:
            return {"output": "Hello!"}

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
        self._invoke_stream_handler: InvokeStreamHandler | None = None

    @property
    def name(self) -> str:
        """Return the agent name."""
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        """Return agent metadata for discovery."""
        return {
            "capabilities": {
                "streaming": self._invoke_stream_handler is not None,
                **self._metadata.get("capabilities", {}),
            },
            "input": self._metadata.get("input", DEFAULT_AGENT_INPUT),
            "output": self._metadata.get("output", DEFAULT_AGENT_OUTPUT),
            **{
                k: v
                for k, v in self._metadata.items()
                if k not in ("capabilities", "input", "output")
            },
        }

    # Decorator methods for handler registration

    def handler(self, fn: InvokeHandler) -> InvokeHandler:
        """Register a handler.

        Example:
            @agent.handler
            async def handle(request: AgentInvokeRequest) -> AgentInvokeResponseDict:
                return {"output": "Hello!"}
        """
        self._invoke_handler = fn
        return fn

    def stream_handler(self, fn: InvokeStreamHandler) -> InvokeStreamHandler:
        """Register a streaming handler.

        Example:
            @agent.stream_handler
            async def handle(request: AgentInvokeRequest):
                yield "Hello"
                yield " world!"
        """
        self._invoke_stream_handler = fn
        return fn

    # Implementation of abstract methods

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponseDict:
        """Handle an invoke request."""
        if self._invoke_handler is None:
            raise NotImplementedError(f"No invoke handler registered for agent '{self._name}'")
        return await self._invoke_handler(request)

    async def invoke_stream(self, request: AgentInvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request."""
        if self._invoke_stream_handler is None:
            raise NotImplementedError(
                f"No streaming invoke handler registered for agent '{self._name}'"
            )
        async for chunk in self._invoke_stream_handler(request):
            yield chunk


def _extract_input_from_function(
    func: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Extract JSON Schema input and output from function signature.

    Parses docstrings (Google, NumPy, or Sphinx style) to extract:
    - Parameter descriptions from Args section
    - Return description from Returns section

    Returns:
        A tuple of (input_schema, output_schema).
        input_schema is a dict with 'type', 'properties', and 'required' keys.
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


def agent(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    template: AgentTemplate | None = None,
) -> Agent | Callable[[Callable[..., Any]], Agent]:
    """Decorator to create an agent from a function.

    The function parameters become the agent's input schema (like @tool),
    unless template is set, in which case the template's input/output schema is used.

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

        @agent(template="chat")
        async def chat_handler(messages: list):
            '''Reply to messages.'''
            last = messages[-1] if messages else {}
            return f"Reply to: {last.get('content', '')}"

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
        template: Optional template (prompt, chat, task). When set, uses that template's
            input/output schema instead of deriving from the function signature.

    Returns:
        An Agent instance that handles invoke requests.
    """

    def decorator(f: Callable[..., Any]) -> Agent:
        agent_name = name or f.__name__
        agent_description = description or (f.__doc__ or "").strip().split("\n\n")[0].strip()

        # Detect if streaming (async generator function)
        is_streaming = inspect.isasyncgenfunction(f)

        # Resolve input and output schemas: template overrides derivation
        if template is not None and template in AGENT_TEMPLATES:
            t = AGENT_TEMPLATES[template]
            input_schema = t["input"]
            output_schema = t["output"]
        else:
            input_schema, output_schema = _extract_input_from_function(f)
            output_schema = output_schema or DEFAULT_AGENT_OUTPUT

        # Build metadata
        metadata: dict[str, Any] = {
            "description": agent_description,
            "input": input_schema,
            "output": output_schema,
            "capabilities": {"streaming": is_streaming},
        }
        if template is not None:
            metadata["template"] = template

        # Create agent instance
        agent_instance = Agent(
            agent_name,
            metadata=metadata,
        )

        if is_streaming:
            # Register streaming invoke handler
            async def invoke_stream_handler(request: AgentInvokeRequest) -> AsyncIterator[str]:
                async for chunk in f(**request.input):
                    # Convert to string if not already
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        yield json.dumps(chunk)

            agent_instance.stream_handler(invoke_stream_handler)

            # Also register non-streaming handler that collects chunks
            async def invoke_handler(request: AgentInvokeRequest) -> AgentInvokeResponseDict:
                chunks: list[str] = []
                async for chunk in f(**request.input):
                    if isinstance(chunk, str):
                        chunks.append(chunk)
                    else:
                        chunks.append(str(chunk))
                return {"output": "".join(chunks)}

            agent_instance.handler(invoke_handler)
        else:
            # Register regular invoke handler
            async def invoke_handler(request: AgentInvokeRequest) -> AgentInvokeResponseDict:
                if inspect.iscoroutinefunction(f):
                    result = await f(**request.input)
                else:
                    result = f(**request.input)
                return {"output": result}

            agent_instance.handler(invoke_handler)

        # Preserve function metadata
        agent_instance.__doc__ = f.__doc__
        agent_instance.__name__ = f.__name__  # type: ignore[attr-defined]
        agent_instance.__module__ = f.__module__

        return agent_instance

    # Handle both @agent and @agent(...) syntax
    if func is not None:
        return decorator(func)

    return decorator
