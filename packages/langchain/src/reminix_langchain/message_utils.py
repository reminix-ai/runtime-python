"""Message conversion utilities between Reminix and LangChain formats."""

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from reminix_runtime import Message, message_content_to_text


def to_langchain_message(message: Message) -> BaseMessage:
    """Convert a Reminix message to a LangChain message."""
    role = message.role
    content = message_content_to_text(message.content)

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function["name"],
                    "args": json.loads(tc.function["arguments"]),
                    "type": "tool_call",
                }
                for tc in message.tool_calls
            ]
        return AIMessage(
            content=content,
            **({"tool_calls": tool_calls} if tool_calls else {}),
        )
    elif role in ("system", "developer"):
        return SystemMessage(content=content)
    elif role == "tool":
        tool_call_id = getattr(message, "tool_call_id", None) or "unknown"
        return ToolMessage(content=content, tool_call_id=tool_call_id)
    else:
        return HumanMessage(content=content)


def from_langchain_message(lc_message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain BaseMessage to a Reminix message dict."""
    msg_type = lc_message.type
    content = (
        lc_message.content
        if isinstance(lc_message.content, str)
        else json.dumps(lc_message.content)
    )

    if msg_type == "human":
        return {"role": "user", "content": content}
    elif msg_type == "ai":
        result: dict[str, Any] = {"role": "assistant", "content": content}
        ai_msg: Any = lc_message
        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.get("id", "unknown"),
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc.get("args", {})),
                    },
                }
                for tc in ai_msg.tool_calls
            ]
        return result
    elif msg_type == "system":
        return {"role": "system", "content": content}
    elif msg_type == "tool":
        tool_msg: Any = lc_message
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": getattr(tool_msg, "tool_call_id", "unknown"),
        }
    else:
        return {"role": "user", "content": content}
