"""LlamaIndex adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from reminix_runtime import (
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
    Message,
)


@runtime_checkable
class ChatEngine(Protocol):
    """Protocol for LlamaIndex chat engines."""

    async def achat(self, message: str) -> Any:
        """Async chat method."""
        ...

    async def astream_chat(self, message: str) -> Any:
        """Async streaming chat method."""
        ...


class LlamaIndexAdapter(BaseAdapter):
    """Adapter for LlamaIndex chat engines."""

    adapter_name = "llamaindex"

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
                return message.content or ""
        # Fallback to last message if no user message found
        return messages[-1].content or "" if messages else ""

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request.

        For task-oriented operations. Expects input with 'query' or 'prompt' key.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        # Extract query from input
        if "query" in request.input:
            query = request.input["query"]
        elif "prompt" in request.input:
            query = request.input["prompt"]
        elif "message" in request.input:
            query = request.input["message"]
        else:
            query = str(request.input)

        # Call the chat engine
        response = await self._engine.achat(query)

        # Extract content from response
        output = str(response.response) if hasattr(response, "response") else str(response)

        return InvokeResponse(output=output)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        For conversational interactions. Sends the last user message to the engine.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with output and messages.
        """
        # Get the last user message to send to the engine
        message = self._get_last_user_message(request.messages)

        # Call the chat engine
        response = await self._engine.achat(message)

        # Extract content from response
        output = str(response.response) if hasattr(response, "response") else str(response)

        # Build response messages (original + assistant response)
        response_messages: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append({"role": "assistant", "content": output})

        return ChatResponse(output=output, messages=response_messages)

    async def invoke_stream(self, request: InvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request.

        Args:
            request: The invoke request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        # Extract query from input
        if "query" in request.input:
            query = request.input["query"]
        elif "prompt" in request.input:
            query = request.input["prompt"]
        elif "message" in request.input:
            query = request.input["message"]
        else:
            query = str(request.input)

        # Stream from the chat engine
        response = await self._engine.astream_chat(query)
        async for token in response.async_response_gen():
            yield json.dumps({"chunk": token})

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request.

        Args:
            request: The chat request with messages.

        Yields:
            JSON-encoded chunks from the stream.
        """
        # Get the last user message to send to the engine
        message = self._get_last_user_message(request.messages)

        # Stream from the chat engine
        response = await self._engine.astream_chat(message)
        async for token in response.async_response_gen():
            yield json.dumps({"chunk": token})


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
