"""Reminix Runtime Types."""

from typing import Literal, Any

from pydantic import BaseModel, Field


# Valid message roles
Role = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """A message in the conversation."""

    role: Role
    content: str


class InvokeRequest(BaseModel):
    """Request for invoke endpoint."""

    messages: list[Message] = Field(..., min_length=1)
    context: dict[str, Any] | None = None


class InvokeResponse(BaseModel):
    """Response from invoke endpoint."""

    content: str
    messages: list[dict[str, Any]]


class ChatRequest(BaseModel):
    """Request for chat endpoint."""

    messages: list[Message] = Field(..., min_length=1)
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    content: str
    messages: list[dict[str, Any]]
