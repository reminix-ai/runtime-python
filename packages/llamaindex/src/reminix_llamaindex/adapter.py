"""LlamaIndex adapter for Reminix Runtime."""

from typing import Any, Protocol, runtime_checkable

from reminix_runtime import (
    BaseAdapter,
    InvokeRequest,
    InvokeResponse,
    ChatRequest,
    ChatResponse,
    Message,
)


@runtime_checkable
class ChatEngine(Protocol):
    """Protocol for LlamaIndex chat engines."""

    async def achat(self, message: str) -> Any:
        """Async chat method."""
        ...


class LlamaIndexAdapter(BaseAdapter):
    """Adapter for LlamaIndex chat engines."""

    def __init__(self, engine: ChatEngine, name: str = "llamaindex-agent") -> None:
        """Initialize the adapter.

        Args:
            engine: A LlamaIndex chat engine (e.g., SimpleChatEngine, ContextChatEngine).
            name: Name for the agent.
        """
        self._engine = engine
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _get_last_user_message(self, messages: list[Message]) -> str:
        """Get the last user message from the conversation."""
        for message in reversed(messages):
            if message.role == "user":
                return message.content
        # Fallback to last message if no user message found
        return messages[-1].content if messages else ""

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request.

        Args:
            request: The invoke request with messages.

        Returns:
            The invoke response with the engine's reply.
        """
        # Get the last user message to send to the engine
        message = self._get_last_user_message(request.messages)

        # Call the chat engine
        response = await self._engine.achat(message)

        # Extract content from response
        content = str(response.response) if hasattr(response, "response") else str(response)

        # Build response messages (original + assistant response)
        response_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append({"role": "assistant", "content": content})

        return InvokeResponse(content=content, messages=response_messages)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with the engine's reply.
        """
        # Get the last user message to send to the engine
        message = self._get_last_user_message(request.messages)

        # Call the chat engine
        response = await self._engine.achat(message)

        # Extract content from response
        content = str(response.response) if hasattr(response, "response") else str(response)

        # Build response messages (original + assistant response)
        response_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append({"role": "assistant", "content": content})

        return ChatResponse(content=content, messages=response_messages)


def wrap(engine: ChatEngine, name: str = "llamaindex-agent") -> LlamaIndexAdapter:
    """Wrap a LlamaIndex chat engine for use with Reminix Runtime.

    Args:
        engine: A LlamaIndex chat engine (e.g., SimpleChatEngine, ContextChatEngine).
        name: Name for the agent.

    Returns:
        A LlamaIndexAdapter instance.

    Example:
        ```python
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.llms.openai import OpenAI
        from reminix_llamaindex import wrap
        from reminix_runtime import serve

        llm = OpenAI(model="gpt-4")
        engine = SimpleChatEngine.from_defaults(llm=llm)
        agent = wrap(engine, name="my-agent")
        serve([agent], port=8080)
        ```
    """
    return LlamaIndexAdapter(engine, name=name)
