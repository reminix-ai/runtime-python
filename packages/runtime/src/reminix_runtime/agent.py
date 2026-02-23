"""Agent for Reminix Runtime."""

import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from .schemas import AGENT_TYPES, DEFAULT_AGENT_INPUT, DEFAULT_AGENT_OUTPUT, AgentType
from .tool import _extract_schema_from_function
from .types import AgentRequest

# === Agent Base Class ===


class Agent:
    """Base class for all agents.

    Framework agents extend this class. The agent() factory creates
    a private _FunctionAgent subclass internally.
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        streaming: bool = False,
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        type: AgentType | None = None,
        framework: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._streaming = streaming
        self._input_schema = input_schema or DEFAULT_AGENT_INPUT
        self._output_schema = output_schema or DEFAULT_AGENT_OUTPUT
        self._type = type
        self._framework = framework
        self.instructions = instructions
        self._tags = tags
        self._extra_metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "description": self._description,
            "capabilities": {"streaming": self._streaming},
            "inputSchema": self._input_schema,
            "outputSchema": self._output_schema,
        }
        if self._type:
            result["type"] = self._type
        if self._framework:
            result["framework"] = self._framework
        if self._tags:
            result["tags"] = self._tags
        if self._extra_metadata:
            result.update(self._extra_metadata)
        return result

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        raise NotImplementedError

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:  # noqa: ARG002
        raise NotImplementedError
        yield  # unreachable; makes this an async generator


# === _FunctionAgent (private) ===


class _FunctionAgent(Agent):
    """Agent created by the agent() factory from a function."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        streaming: bool,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        type: AgentType | None,
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        invoke_fn: Callable[[AgentRequest], Awaitable[dict[str, Any]]],
        invoke_stream_fn: Callable[[AgentRequest], AsyncIterator[str]] | None = None,
    ):
        super().__init__(
            name=name,
            description=description,
            streaming=streaming,
            input_schema=input_schema,
            output_schema=output_schema,
            type=type,
            tags=tags,
            metadata=metadata,
        )
        self._invoke_fn = invoke_fn
        self._invoke_stream_fn = invoke_stream_fn

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        return await self._invoke_fn(request)

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        if self._invoke_stream_fn is None:
            raise NotImplementedError(f"Streaming not supported for agent '{self._name}'")
        async for chunk in self._invoke_stream_fn(request):
            yield chunk


# === agent() factory ===


def agent(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    type: AgentType | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Agent | Callable[[Callable[..., Any]], Agent]:
    """Decorator to create an agent from a function.

    The function parameters become the agent's input schema (like @tool),
    unless type is set, in which case the type's input/output schema is used.

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

        @agent(type="chat")
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
        type: Optional agent type (prompt, chat, task, rag, thread, workflow).
            When set, uses that type's input/output schema instead of
            deriving from the function signature.
        tags: Optional list of tags for categorization.
        metadata: Optional extra metadata to include in the agent's metadata.

    Returns:
        An Agent instance that handles invoke requests.
    """

    def _call_with_request(f: Callable[..., Any], request: AgentRequest) -> dict[str, Any]:
        """Build kwargs for f: **request.input and optionally context=request.context."""
        kwargs = dict(request.input)
        sig = inspect.signature(f)
        if "context" in sig.parameters:
            kwargs["context"] = request.context
        return kwargs

    def decorator(f: Callable[..., Any]) -> Agent:
        agent_name = name or f.__name__
        agent_description = description or (f.__doc__ or "").strip().split("\n\n")[0].strip()

        # Detect if streaming (async generator function)
        is_streaming = inspect.isasyncgenfunction(f)

        # Resolve input and output schemas: type overrides derivation
        if type is not None and type in AGENT_TYPES:
            t = AGENT_TYPES[type]
            input_schema = t["inputSchema"]
            output_schema = t["outputSchema"]
        else:
            _, input_schema, output_schema = _extract_schema_from_function(
                f, skip_params={"messages", "context"}
            )
            output_schema = output_schema or DEFAULT_AGENT_OUTPUT

        # Build handler functions
        invoke_fn: Callable[[AgentRequest], Awaitable[dict[str, Any]]]
        invoke_stream_fn: Callable[[AgentRequest], AsyncIterator[str]] | None = None

        if is_streaming:

            async def _invoke_stream(request: AgentRequest) -> AsyncIterator[str]:
                kwargs = _call_with_request(f, request)
                async for chunk in f(**kwargs):
                    yield str(chunk) if not isinstance(chunk, str) else chunk

            invoke_stream_fn = _invoke_stream

            async def _invoke_collecting(request: AgentRequest) -> dict[str, Any]:
                kwargs = _call_with_request(f, request)
                chunks: list[str] = []
                async for chunk in f(**kwargs):
                    chunks.append(str(chunk) if not isinstance(chunk, str) else chunk)
                return {"output": "".join(chunks)}

            invoke_fn = _invoke_collecting
        else:

            async def _invoke(request: AgentRequest) -> dict[str, Any]:
                kwargs = _call_with_request(f, request)
                if inspect.iscoroutinefunction(f):
                    result = await f(**kwargs)
                else:
                    result = f(**kwargs)
                return {"output": result}

            invoke_fn = _invoke

        agent_instance = _FunctionAgent(
            name=agent_name,
            description=agent_description,
            streaming=is_streaming,
            input_schema=input_schema,
            output_schema=output_schema,
            type=type,
            tags=tags,
            metadata=metadata,
            invoke_fn=invoke_fn,
            invoke_stream_fn=invoke_stream_fn,
        )

        # Preserve function metadata
        agent_instance.__doc__ = f.__doc__
        agent_instance.__name__ = f.__name__  # type: ignore[attr-defined]
        agent_instance.__module__ = f.__module__

        return agent_instance

    # Handle both @agent and @agent(...) syntax
    if func is not None:
        return decorator(func)

    return decorator
