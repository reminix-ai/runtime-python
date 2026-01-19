"""Reminix Runtime Tool definitions."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, get_type_hints

from .types import ToolExecuteRequest, ToolExecuteResponse, ToolSchema


class ToolBase(ABC):
    """Abstract base class for all tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> ToolSchema:
        """JSON Schema for the tool's input parameters."""
        ...

    @property
    def output(self) -> dict[str, Any] | None:
        """Optional JSON Schema for the tool's output."""
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata for runtime discovery."""
        meta = {
            "type": "tool",
            "description": self.description,
            "parameters": self.parameters.model_dump(),
        }
        if self.output:
            meta["output"] = self.output
        return meta

    @abstractmethod
    async def execute(self, request: ToolExecuteRequest) -> ToolExecuteResponse:
        """Execute the tool with the given input."""
        ...


# Type mapping from Python types to JSON Schema types
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


def _extract_schema_from_function(
    func: Callable[..., Any],
) -> tuple[str, ToolSchema, dict[str, Any] | None]:
    """Extract tool schema from function signature and docstring.

    Returns:
        Tuple of (description, parameters_schema, output_schema)
    """
    # Get description from docstring
    description = func.__doc__ or f"Tool: {func.__name__}"
    # Clean up docstring - take first line/paragraph
    description = description.strip().split("\n\n")[0].strip()

    # Get type hints
    hints = get_type_hints(func)
    return_type = hints.pop("return", None)  # Extract return type hint
    output_schema = _python_type_to_json_schema(return_type) if return_type else None

    # Get signature for defaults
    sig = inspect.signature(func)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip self, cls, request, context parameters
        if param_name in ("self", "cls", "request", "context"):
            continue

        # Get type hint
        param_type = hints.get(param_name, str)
        schema = _python_type_to_json_schema(param_type)

        # Add description if available (could parse from docstring Args section)
        properties[param_name] = schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            # Add default value to schema
            properties[param_name]["default"] = param.default

    return description, ToolSchema(properties=properties, required=required), output_schema


class Tool(ToolBase):
    """A tool created from a function using the @tool decorator."""

    def __init__(
        self,
        func: Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ):
        """Create a tool from a function.

        Args:
            func: The function to wrap as a tool.
            name: Optional name override. Defaults to function name.
            description: Optional description override. Defaults to docstring.
        """
        self._func = func
        self._name = name or func.__name__
        extracted_desc, self._parameters, self._output = _extract_schema_from_function(func)
        self._description = description or extracted_desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> ToolSchema:
        return self._parameters

    @property
    def output(self) -> dict[str, Any] | None:
        return self._output

    async def execute(self, request: ToolExecuteRequest) -> ToolExecuteResponse:
        """Execute the tool by calling the wrapped function."""
        try:
            # Check if function is async
            if inspect.iscoroutinefunction(self._func):
                result = await self._func(**request.input)
            else:
                result = self._func(**request.input)

            return ToolExecuteResponse(output=result)
        except Exception as e:
            return ToolExecuteResponse(output=None, error=str(e))


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Decorator to create a tool from a function.

    Can be used with or without arguments. The return type hint is automatically
    extracted and included as the output schema in the tool's metadata.

        @tool
        async def my_tool(param: str) -> dict[str, str]:
            '''Tool description.'''
            return {"result": param}

        @tool(name="custom-name", description="Custom description")
        async def my_tool(param: str) -> dict:
            return {"result": param}

    Args:
        func: The function to wrap (when used without parentheses).
        name: Optional name override. Defaults to function name.
        description: Optional description override. Defaults to docstring.

    Returns:
        A Tool instance with parameters and output schemas extracted from type hints.
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        tool_instance = Tool(f, name=name, description=description)
        # Preserve function metadata
        tool_instance.__doc__ = f.__doc__
        tool_instance.__name__ = f.__name__  # type: ignore[attr-defined]
        tool_instance.__module__ = f.__module__
        return tool_instance

    # Handle both @tool and @tool(...) syntax
    if func is not None:
        return decorator(func)

    return decorator
