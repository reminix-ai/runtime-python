"""Reminix Runtime Tool definitions."""

import inspect
from collections.abc import Callable
from typing import (
    Any,
    get_args,
    get_origin,
    get_type_hints,
    is_typeddict,
)

from docstring_parser import parse as parse_docstring
from pydantic import BaseModel

from .types import ToolRequest

# === Shared Schema Utilities ===


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
    *,
    skip_params: set[str] | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    """Extract description, input schema, and output schema from a function.

    Parses docstrings (Google, NumPy, or Sphinx style) to extract:
    - Short description (first line/paragraph)
    - Parameter descriptions from Args section
    - Return description from Returns section

    Args:
        func: The function to extract schema from.
        skip_params: Additional parameter names to skip (beyond self, cls).

    Returns:
        Tuple of (description, input_schema, output_schema).
    """
    always_skip = {"self", "cls"}
    skip = always_skip | (skip_params or set())

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
        if param_name in skip:
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


# === Tool Base Class ===


class Tool:
    """Base class for all tools.

    The tool() factory creates a private _FunctionTool subclass internally.
    """

    def __init__(
        self,
        name: str,
        *,
        description: str = "",
        input_schema: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._input_schema = input_schema or {"type": "object", "properties": {}, "required": []}
        self._output_schema = output_schema or {"type": "string"}
        self._tags = tags
        self._extra_metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "description": self._description,
            "inputSchema": self._input_schema,
            "outputSchema": self._output_schema,
        }
        if self._tags:
            result["tags"] = self._tags
        if self._extra_metadata:
            result.update(self._extra_metadata)
        return result

    async def call(self, request: ToolRequest) -> dict[str, Any]:
        raise NotImplementedError


# === _FunctionTool (private) ===


class _FunctionTool(Tool):
    """Tool created by the tool() factory from a function."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        tags: list[str] | None,
        metadata: dict[str, Any] | None,
        call_fn: Callable[[ToolRequest], Any],
    ):
        super().__init__(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema,
            tags=tags,
            metadata=metadata,
        )
        self._call_fn = call_fn

    async def call(self, request: ToolRequest) -> dict[str, Any]:
        return await self._call_fn(request)


# === tool() factory ===


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
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
        tags: Optional list of tags for categorization.
        metadata: Optional extra metadata to include in the tool's metadata.

    Returns:
        A Tool instance with input and output schemas extracted from type hints.
    """

    def decorator(f: Callable[..., Any]) -> Tool:
        tool_name = name or f.__name__
        extracted_desc, input_schema, output_schema = _extract_schema_from_function(
            f, skip_params={"context"}
        )
        tool_description = description or extracted_desc

        async def call_fn(request: ToolRequest) -> dict[str, Any]:
            kwargs = dict(request.arguments)
            sig = inspect.signature(f)
            if "context" in sig.parameters:
                kwargs["context"] = request.context
            if inspect.iscoroutinefunction(f):
                result = await f(**kwargs)
            else:
                result = f(**kwargs)
            return {"output": result}

        tool_instance = _FunctionTool(
            name=tool_name,
            description=tool_description,
            input_schema=input_schema,
            output_schema=output_schema or {"type": "string"},
            tags=tags,
            metadata=metadata,
            call_fn=call_fn,
        )

        # Preserve function metadata
        tool_instance.__doc__ = f.__doc__
        tool_instance.__name__ = f.__name__  # type: ignore[attr-defined]
        tool_instance.__module__ = f.__module__

        return tool_instance

    # Handle both @tool and @tool(...) syntax
    if func is not None:
        return decorator(func)

    return decorator
