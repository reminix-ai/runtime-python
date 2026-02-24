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


@runtime_checkable
class QueryEngine(Protocol):
    """Protocol for LlamaIndex query engines."""

    async def aquery(self, query: str) -> Any: ...


def _is_chat_engine(engine: Any) -> bool:
    """Detect if the engine is a chat engine (has achat)."""
    return hasattr(engine, "achat") and callable(engine.achat)


class LlamaIndexRagAgent(Agent):
    """LlamaIndex RAG agent for chat engines and query engines."""

    def __init__(
        self,
        engine: ChatEngine | QueryEngine,
        *,
        name: str = "llamaindex-agent",
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._is_chat = _is_chat_engine(engine)
        super().__init__(
            name,
            description=description or "llamaindex rag agent",
            streaming=self._is_chat,
            input_schema=AGENT_TYPES["rag"]["inputSchema"],
            output_schema=AGENT_TYPES["rag"]["outputSchema"],
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

    def _extract_response_text(self, response: Any) -> str:
        """Extract text from a LlamaIndex response object."""
        return str(response.response) if hasattr(response, "response") else str(response)

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        query = self._extract_query(request)
        if self.instructions:
            query = f"{self.instructions}\n\n{query}"

        if self._is_chat:
            response = await self._engine.achat(query)  # type: ignore[union-attr]
        else:
            response = await self._engine.aquery(query)  # type: ignore[union-attr]

        output = self._extract_response_text(response)
        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        query = self._extract_query(request)
        if self.instructions:
            query = f"{self.instructions}\n\n{query}"

        if self._is_chat:
            response = await self._engine.astream_chat(query)  # type: ignore[union-attr]
            async for token in response.async_response_gen():
                yield token
        else:
            # Query engines don't support token streaming — yield the full response.
            response = await self._engine.aquery(query)  # type: ignore[union-attr]
            yield self._extract_response_text(response)
