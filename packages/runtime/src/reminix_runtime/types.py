"""Reminix Runtime Types."""

from pydantic import BaseModel


class Message(BaseModel):
    """A message in the conversation."""

    role: str
    content: str


class InvokeRequest(BaseModel):
    """Request for invoke endpoint."""

    messages: list[Message]


class InvokeResponse(BaseModel):
    """Response from invoke endpoint."""

    content: str
    messages: list[dict]


class ChatRequest(BaseModel):
    """Request for chat endpoint."""

    messages: list[Message]


class ChatResponse(BaseModel):
    """Response from chat endpoint."""

    content: str
    messages: list[dict]
