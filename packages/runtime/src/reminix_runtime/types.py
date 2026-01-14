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


class InvokeRequest(BaseModel):
    """Request for invoke endpoint."""

    input: dict[str, Any] = Field(..., min_length=1)
    stream: bool = False
    context: dict[str, Any] | None = None


class InvokeResponse(BaseModel):
    """Response from invoke endpoint."""

    output: Any


class ChatRequest(BaseModel):
    """Request for chat endpoint."""

    messages: list[Message] = Field(..., min_length=1)
    stream: bool = False
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    output: str
    messages: list[dict[str, Any]]
