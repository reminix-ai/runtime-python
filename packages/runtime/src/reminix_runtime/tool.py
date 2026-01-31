"""Reminix Runtime Tool definitions."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints, is_typeddict

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel

from .types import ToolCallRequest, ToolCallResponse

# Default output schema for tools
# Response: { "output": "..." }
DEFAULT_TOOL_OUTPUT: dict[str, Any] = {
    "type": "string",
}


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
    def input(self) -> dict[str, Any]:
        """JSON Schema for the tool's input."""
        ...

    @property
    def output(self) -> dict[str, Any] | None:
        """Optional JSON Schema for the tool's output."""
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        """Metadata for runtime discovery."""
        meta = {
            "description": self.description,
            "input": self.input,
        }
        if self.output:
            meta["output"] = self.output
        return meta

    @abstractmethod
    async def call(self, request: ToolCallRequest) -> ToolCallResponse:
        """Call the tool with the given input."""
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
    """Convert a Python type hint to a JSON Schema type.

    Supports:
    - Basic types (str, int, float, bool)
    - Pydantic BaseModel subclasses
    - TypedDict classes
    - Generic types (list[T], dict[K, V], Optional[T])
    """
    # Handle None type
    if python_type is type(None):
        return {"type": "null"}

    # Handle Pydantic models
    if isinstance(python_type, type) and issubclass(python_type, BaseModel):
        # Use Pydantic's built-in JSON schema generation
        schema = python_type.model_json_schema()
        # Remove $defs if present (inline the schema)
        if "$defs" in schema:
            del schema["$defs"]
        return schema

    # Handle TypedDict
    if is_typeddict(python_type):
        annotations = get_type_hints(python_type)
        properties = {}
        for key, value_type in annotations.items():
            properties[key] = _python_type_to_json_schema(value_type)

        # Get required keys (TypedDict tracks these)
        required_keys = list(getattr(python_type, "__required_keys__", set()))

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required_keys:
            schema["required"] = required_keys
        return schema

    # Handle basic types
    if python_type in _TYPE_MAP:
        return {"type": _TYPE_MAP[python_type]}

    # Handle generic types (Optional, list, dict, etc.)
    origin = get_origin(python_type)
    if origin is not None:
        args = get_args(python_type)

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
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Extract tool schema from function signature and docstring.

    Parses docstrings (Google, NumPy, or Sphinx style) to extract:
    - Short description (first line/paragraph)
    - Parameter descriptions from Args section
    - Return description from Returns section

    Returns:
        Tuple of (description, input_schema, output_schema)
    """
    # Parse docstring using docstring-parser (supports Google, NumPy, Sphinx styles)
    docstring = func.__doc__ or ""
    parsed_doc = parse_docstring(docstring)

    # Get description from parsed docstring
    description = parsed_doc.short_description or f"Tool: {func.__name__}"

    # Build parameter descriptions lookup from docstring Args section
    param_descriptions: dict[str, str] = {}
    for param in parsed_doc.params:
        if param.description:
            param_descriptions[param.arg_name] = param.description

    # Get type hints
    hints = get_type_hints(func)
    return_type = hints.pop("return", None)  # Extract return type hint
    output_schema = _python_type_to_json_schema(return_type) if return_type else None

    # Add return description to output schema if available
    if output_schema and parsed_doc.returns and parsed_doc.returns.description:
        output_schema["description"] = parsed_doc.returns.description

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

        # Add description from docstring if available
        if param_name in param_descriptions:
            schema["description"] = param_descriptions[param_name]

        properties[param_name] = schema

        # Check if required (no default value)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            # Add default value to schema
            properties[param_name]["default"] = param.default

    input_schema = {"type": "object", "properties": properties, "required": required}
    return description, input_schema, output_schema


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
        extracted_desc, self._input, self._output = _extract_schema_from_function(func)
        self._description = description or extracted_desc

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input(self) -> dict[str, Any]:
        return self._input

    @property
    def output(self) -> dict[str, Any] | None:
        return self._output or DEFAULT_TOOL_OUTPUT

    async def call(self, request: ToolCallRequest) -> ToolCallResponse:
        """Call the tool by invoking the wrapped function.

        Exceptions are not caught here - they propagate to the server
        which returns appropriate HTTP error codes.
        """
        # Check if function is async
        if inspect.iscoroutinefunction(self._func):
            result = await self._func(**request.input)
        else:
            result = self._func(**request.input)

        return ToolCallResponse(output=result)


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
        A Tool instance with input and output schemas extracted from type hints.
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
