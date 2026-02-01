"""Reminix Runtime Types."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# Valid message roles (OpenAI current; no legacy function role)
Role = Literal["developer", "system", "user", "assistant", "tool"]


class ToolCall(BaseModel):
    """A single tool call (OpenAI-style)."""

    id: str
    type: Literal["function"] = "function"
    function: dict[str, Any]  # {"name": str, "arguments": str}


# --- Content parts (discriminated by type) ---


class TextContentPart(BaseModel):
    """Content part: text."""

    type: Literal["text"] = "text"
    text: str


class ImageUrlPart(BaseModel):
    """Image URL with optional detail."""

    url: str
    detail: Literal["auto", "low", "high"] = "auto"


class ImageUrlContentPart(BaseModel):
    """Content part: image URL."""

    type: Literal["image_url"] = "image_url"
    image_url: ImageUrlPart


class InputAudioPart(BaseModel):
    """Input audio (base64 + format)."""

    data: str
    format: Literal["wav", "mp3"]


class InputAudioContentPart(BaseModel):
    """Content part: input audio."""

    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudioPart


class FilePart(BaseModel):
    """File (file_id or filename/file_data)."""

    file_id: str | None = None
    filename: str | None = None
    file_data: str | None = None


class FileContentPart(BaseModel):
    """Content part: file."""

    type: Literal["file"] = "file"
    file: FilePart


class RefusalContentPart(BaseModel):
    """Content part: refusal (assistant)."""

    type: Literal["refusal"] = "refusal"
    refusal: str


ContentPart = Annotated[
    TextContentPart
    | ImageUrlContentPart
    | InputAudioContentPart
    | FileContentPart
    | RefusalContentPart,
    Field(discriminator="type"),
]

# Type for list of content parts (used in Message.content)
ContentPartList = list[
    TextContentPart
    | ImageUrlContentPart
    | InputAudioContentPart
    | FileContentPart
    | RefusalContentPart
]


class Message(BaseModel):
    """A message in the conversation (OpenAI-style; input and output)."""

    role: Role
    content: str | ContentPartList | None = None
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
