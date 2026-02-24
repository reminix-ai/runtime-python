"""Message conversion utilities between Reminix and Anthropic formats."""

from typing import Any

from reminix_runtime import Message, message_content_to_text


def to_anthropic_messages(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert Reminix messages to Anthropic format, separating the system message.

    Returns a tuple of (system_message, anthropic_messages).
    """
    system_message: str | None = None
    anthropic_messages: list[dict[str, Any]] = []

    for message in messages:
        text = message_content_to_text(message.content)
        if message.role == "system":
            system_message = text
        elif message.role in ("user", "assistant"):
            anthropic_messages.append({"role": message.role, "content": text})
        elif message.role == "tool":
            anthropic_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id or "",
                            "content": text,
                        }
                    ],
                }
            )

    return system_message, anthropic_messages
