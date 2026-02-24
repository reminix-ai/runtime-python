"""Structured streaming event types for Reminix Runtime.

These are the application-level events yielded during streaming.
Transport-level concerns (done/error) are handled by the server and SDK.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from .types import Message, ToolCall


class TextDeltaEvent(BaseModel):
    """A text token chunk."""

    type: Literal["text_delta"] = "text_delta"
    delta: str


class ToolCallEvent(BaseModel):
    """An assistant tool call."""

    type: Literal["tool_call"] = "tool_call"
    tool_call: ToolCall


class ToolResultEvent(BaseModel):
    """The result of a tool call."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    output: str


class MessageEvent(BaseModel):
    """A complete message (used by thread/graph agents)."""

    type: Literal["message"] = "message"
    message: Message


class PendingAction(BaseModel):
    """Pending action for a paused workflow step."""

    step: str
    type: str
    message: str
    options: list[str] | None = None


class StepEvent(BaseModel):
    """A workflow step completion (used by workflow agents)."""

    type: Literal["step"] = "step"
    name: str
    status: str
    output: Any | None = None
    pendingAction: PendingAction | None = Field(default=None, alias="pendingAction")


StreamEvent = Annotated[
    TextDeltaEvent | ToolCallEvent | ToolResultEvent | MessageEvent | StepEvent,
    Field(discriminator="type"),
]
