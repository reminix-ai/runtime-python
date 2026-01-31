"""Reminix Runtime Types."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Valid message roles
Role = Literal["user", "assistant", "system", "tool"]


class ToolCall(BaseModel):
    """A single tool call (OpenAI-style)."""

    id: str
    type: Literal["function"] = "function"
    function: dict[str, Any]  # {"name": str, "arguments": str}


class Message(BaseModel):
    """A message in the conversation (OpenAI-style; supports tool_calls and tool results)."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


# === Request Types ===


class InvokeRequest(BaseModel):
    """Base request type for invoke/call operations."""

    input: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    context: dict[str, Any] | None = None


# Semantic type aliases for agent invoke operations
AgentInvokeRequest = InvokeRequest
"""Request type for agent invoke operations."""

# Semantic type aliases for tool call operations
ToolCallRequest = InvokeRequest
"""Request type for tool call operations."""


# === Response Types ===


class InvokeResponse(BaseModel):
    """Base response type for invoke/call operations."""

    output: Any
    metadata: dict[str, Any] | None = None


# Semantic type aliases for agent invoke operations
AgentInvokeResponse = InvokeResponse
"""Response type for agent invoke operations."""

# Semantic type aliases for tool call operations
ToolCallResponse = InvokeResponse
"""Response type for tool call operations."""

# InvokeResponse as a dict for flexibility
InvokeResponseDict = dict[str, Any]
AgentInvokeResponseDict = InvokeResponseDict
ToolCallResponseDict = InvokeResponseDict


# === Capabilities ===


class Capabilities(BaseModel):
    """Agent/tool capabilities."""

    streaming: bool | None = None
    # batch: bool | None = None      # Process multiple inputs in one call
    # async_: bool | None = None     # Fire-and-forget with webhook callback
    # retry: bool | None = None      # Built-in retry with backoff


# === Error Types ===


class RuntimeError(BaseModel):
    """Structured runtime error information."""

    type: str
    """Error type/category (e.g., 'ValidationError', 'ExecutionError')."""

    message: str
    """Human-readable error message."""

    stack: str | None = None
    """Stack trace (only included when REMINIX_CLOUD is enabled)."""


class RuntimeErrorResponse(BaseModel):
    """Error response from runtime endpoints."""

    error: RuntimeError
