"""LlamaIndex agent adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from reminix_runtime import (
    AgentAdapter,
    ExecuteRequest,
    ExecuteResponse,
    Message,
    serve,
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


class LlamaIndexAgentAdapter(AgentAdapter):
    """Agent adapter for LlamaIndex chat engines."""

    adapter_name = "llamaindex"

    @property
    def metadata(self) -> dict[str, Any]:
        """Return adapter metadata for discovery.

        LlamaIndex adapters accept 'messages', 'prompt', or 'query' inputs.
        """
        return {
            "type": "adapter",
            "adapter": self.adapter_name,
            "requestKeys": ["messages", "prompt", "query"],
            "responseKeys": ["output"],
        }

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

    def _extract_query(self, request: ExecuteRequest) -> str:
        """Extract query string from execute request."""
        # Check if input contains messages (chat-style)
        if "messages" in request.input:
            messages_data = request.input["messages"]
            messages = [Message(**m) if isinstance(m, dict) else m for m in messages_data]
            return self._get_last_user_message(messages)
        elif "query" in request.input:
            return request.input["query"]
        elif "prompt" in request.input:
            return request.input["prompt"]
        elif "message" in request.input:
            return request.input["message"]
        else:
            return str(request.input)

    async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        """Handle an execute request.

        For both task-oriented and chat-style operations. Expects input with 'messages',
        'query', 'prompt', or 'message' key.

        Args:
            request: The execute request with input data.

        Returns:
            The execute response with the output.
        """
        query = self._extract_query(request)

        # Call the chat engine
        response = await self._engine.achat(query)

        # Extract content from response
        output = str(response.response) if hasattr(response, "response") else str(response)

        return {"output": output}

    async def execute_stream(self, request: ExecuteRequest) -> AsyncIterator[str]:
        """Handle a streaming execute request.

        Args:
            request: The execute request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        query = self._extract_query(request)

        # Stream from the chat engine
        response = await self._engine.astream_chat(query)
        async for token in response.async_response_gen():
            yield json.dumps({"chunk": token})


def wrap_agent(engine: ChatEngine, name: str = "llamaindex-agent") -> LlamaIndexAgentAdapter:
    """Wrap a LlamaIndex chat engine for use with Reminix Runtime.

    Args:
        engine: A LlamaIndex chat engine (e.g., SimpleChatEngine, ContextChatEngine).
        name: Name for the agent.

    Returns:
        A LlamaIndexAgentAdapter instance.

    Example:
        ```python
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.llms.openai import OpenAI
        from reminix_llamaindex import wrap_agent
        from reminix_runtime import serve

        llm = OpenAI(model="gpt-4")
        engine = SimpleChatEngine.from_defaults(llm=llm)
        agent = wrap_agent(engine, name="my-agent")
        serve(agents=[agent], port=8080)
        ```
    """
    return LlamaIndexAgentAdapter(engine, name=name)


def serve_agent(
    engine: ChatEngine,
    name: str = "llamaindex-agent",
    port: int = 8080,
    host: str = "0.0.0.0",
) -> None:
    """Wrap a LlamaIndex chat engine and serve it immediately.

    This is a convenience function that combines `wrap` and `serve` for single-agent setups.

    Args:
        engine: A LlamaIndex chat engine (e.g., SimpleChatEngine, ContextChatEngine).
        name: Name for the agent.
        port: Port to serve on.
        host: Host to bind to.

    Example:
        ```python
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.llms.openai import OpenAI
        from reminix_llamaindex import serve_agent

        llm = OpenAI(model="gpt-4")
        engine = SimpleChatEngine.from_defaults(llm=llm)
        serve_agent(engine, name="my-agent", port=8080)
        ```
    """
    agent = wrap_agent(engine, name=name)
    serve(agents=[agent], port=port, host=host)
