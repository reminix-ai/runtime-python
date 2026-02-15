"""LlamaIndex RAG adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from reminix_runtime import (
    AGENT_TEMPLATES,
    AgentRequest,
    Message,
    build_messages_from_input,
    message_content_to_text,
)


@runtime_checkable
class ChatEngine(Protocol):
    """Protocol for LlamaIndex chat engines."""

    async def achat(self, message: str) -> Any: ...
    async def astream_chat(self, message: str) -> Any: ...


class LlamaIndexRagAgent:
    """RAG agent adapter for LlamaIndex chat engines."""

    def __init__(self, engine: ChatEngine, name: str = "llamaindex-agent") -> None:
        self._engine = engine
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "description": "llamaindex adapter",
            "capabilities": {"streaming": True},
            "input": AGENT_TEMPLATES["rag"]["input"],
            "output": {"type": "string"},
            "adapter": "llamaindex",
            "template": "rag",
        }

    def _get_last_user_message(self, messages: list[Message]) -> str:
        """Get the last user message from the conversation."""
        for message in reversed(messages):
            if message.role == "user":
                return message_content_to_text(message.content)
        return message_content_to_text(messages[-1].content) if messages else ""

    def _extract_query(self, request: AgentRequest) -> str:
        """Extract query string from invoke request."""
        if "messages" in request.input:
            messages = build_messages_from_input(request)
            return self._get_last_user_message(messages)
        elif "query" in request.input:
            return str(request.input["query"])
        elif "prompt" in request.input:
            return str(request.input["prompt"])
        elif "message" in request.input:
            return str(request.input["message"])
        else:
            return str(request.input)

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        query = self._extract_query(request)
        response = await self._engine.achat(query)
        output = str(response.response) if hasattr(response, "response") else str(response)
        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        query = self._extract_query(request)
        response = await self._engine.astream_chat(query)
        async for token in response.async_response_gen():
            yield json.dumps({"chunk": token})
