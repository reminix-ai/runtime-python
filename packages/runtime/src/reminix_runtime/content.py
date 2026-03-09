"""Helpers for message content (str | ContentPartList | None)."""

from __future__ import annotations

import json

from .types import AgentRequest, ContentPartList, Message


def message_content_to_text(
    content: str | ContentPartList | None,
) -> str:
    """Normalize message content to a single string for providers that only accept text.

    - str: returned as-is
    - ContentPartList: text parts joined; other parts rendered as [type]
    - None: empty string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for p in content:
        text = getattr(p, "text", None) if not isinstance(p, dict) else p.get("text")
        if isinstance(text, str):
            parts.append(text)
        else:
            ptype = getattr(p, "type", None) if not isinstance(p, dict) else p.get("type")
            parts.append(f"[{ptype or 'part'}]")
    return " ".join(parts)


def build_messages_from_input(request: AgentRequest) -> list[Message]:
    """Extract a list of Messages from an AgentRequest's input dict.

    Handles three input shapes that all agents accept:
    - ``{ "messages": [...] }`` — chat-style, returned as Message list
    - ``{ "prompt": "..." }`` — single prompt, wrapped as a user message
    - anything else — stringified and wrapped as a user message
    """
    if "messages" in request.input:
        messages_data = request.input["messages"]
        return [Message(**m) if isinstance(m, dict) else m for m in messages_data]
    elif "prompt" in request.input:
        return [Message(role="user", content=str(request.input["prompt"]))]
    else:
        return [Message(role="user", content=json.dumps(request.input))]
