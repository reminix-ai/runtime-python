"""LlamaIndex RAG agent for Reminix Runtime."""

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
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


class LlamaIndexRagAgent(Agent):
    """LlamaIndex RAG agent for chat engines."""

    def __init__(
        self,
        engine: ChatEngine,
        *,
        name: str = "llamaindex-agent",
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "llamaindex rag agent",
            streaming=True,
            input_schema=AGENT_TYPES["rag"]["input"],
            output_schema=AGENT_TYPES["rag"]["output"],
            type="rag",
            framework="llamaindex",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._engine = engine

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
        if self.instructions:
            query = f"{self.instructions}\n\n{query}"
        response = await self._engine.achat(query)
        output = str(response.response) if hasattr(response, "response") else str(response)
        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        query = self._extract_query(request)
        if self.instructions:
            query = f"{self.instructions}\n\n{query}"
        response = await self._engine.astream_chat(query)
        async for token in response.async_response_gen():
            yield token
