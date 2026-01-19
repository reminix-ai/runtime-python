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


# Tool types


class ToolParameter(BaseModel):
    """A parameter for a tool."""

    name: str
    type: str
    description: str | None = None
    required: bool = True
    default: Any | None = None


class ToolSchema(BaseModel):
    """Schema for a tool's input parameters."""

    type: Literal["object"] = "object"
    properties: dict[str, Any]
    required: list[str] = Field(default_factory=list)


class ToolExecuteRequest(BaseModel):
    """Request for tool execute endpoint."""

    input: dict[str, Any]
    context: dict[str, Any] | None = None


class ToolExecuteResponse(BaseModel):
    """Response from tool execute endpoint."""

    output: Any
    error: str | None = None
