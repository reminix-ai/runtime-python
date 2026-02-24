"""Message conversion utilities between Reminix and Google Gemini formats."""

from google.genai import types

from reminix_runtime import Message, message_content_to_text


def to_gemini_contents(
    messages: list[Message],
) -> tuple[str | None, list[types.Content]]:
    """Convert Reminix messages to Gemini format, separating the system message.

    Returns a tuple of (system_message, contents).
    """
    system_message: str | None = None
    contents: list[types.Content] = []

    for message in messages:
        text = message_content_to_text(message.content)
        if message.role == "system":
            system_message = text
        elif message.role == "user":
            contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
        elif message.role == "assistant":
            contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))

    return system_message, contents
