"""Helpers for message content (str | ContentPartList | None)."""

from .types import ContentPartList


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
