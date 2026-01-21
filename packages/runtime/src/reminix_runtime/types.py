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


class ExecuteRequest(BaseModel):
    """Request for agent execute endpoint."""

    input: dict[str, Any] = Field(default_factory=dict)
    stream: bool = False
    context: dict[str, Any] | None = None


# ExecuteResponse is now a dict with dynamic keys based on agent's responseKeys
# - Regular agents: { "output": ... }
# - Chat agents: { "message": { "role": "assistant", "content": "..." } }
ExecuteResponse = dict[str, Any]


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
