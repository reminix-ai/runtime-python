"""Reminix Runtime Types."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Valid message roles
Role = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """A message in the conversation."""

    role: Role
    content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None


# === Request Types ===


class InvokeRequest(BaseModel):
    """Request for agent/tool invoke endpoint."""

    input: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    context: dict[str, Any] | None = None


# === Response Types ===


class InvokeResponse(BaseModel):
    """Response from agent/tool invoke endpoint."""

    output: Any
    metadata: dict[str, Any] | None = None


# InvokeResponse as a dict for flexibility
InvokeResponseDict = dict[str, Any]


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
