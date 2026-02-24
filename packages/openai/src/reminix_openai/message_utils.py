"""Message conversion utilities between Reminix and OpenAI formats."""

from typing import Any

from reminix_runtime import Message, message_content_to_text


def to_openai_message(message: Message) -> dict[str, Any]:
    """Convert a Reminix message to OpenAI format."""
    role = message.role
    if role not in ("user", "assistant", "system"):
        role = "user"
    result: dict[str, Any] = {
        "role": role,
        "content": message_content_to_text(message.content) or "",
    }
    if message.tool_calls:
        result["tool_calls"] = message.tool_calls
    if message.tool_call_id:
        result["tool_call_id"] = message.tool_call_id
    if message.name:
        result["name"] = message.name
    return result
