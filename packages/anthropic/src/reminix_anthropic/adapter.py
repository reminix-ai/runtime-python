"""Anthropic adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from reminix_runtime import (
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
    Message,
)


class AnthropicAdapter(BaseAdapter):
    """Adapter for Anthropic messages API."""

    adapter_name = "anthropic"

    def __init__(
        self,
        client: AsyncAnthropic,
        name: str = "anthropic-agent",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
    ) -> None:
        """Initialize the adapter.

        Args:
            client: An Anthropic async client.
            name: Name for the agent.
            model: The model to use for completions.
            max_tokens: Maximum tokens in the response.
        """
        self._client = client
        self._name = name
        self._model = model
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def _extract_system_and_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system message and convert remaining messages to Anthropic format.

        Anthropic expects system message as a separate parameter, not in the messages list.

        Returns:
            Tuple of (system_message, messages_list)
        """
        system_message: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system":
                # Anthropic only supports one system message, use the last one
                system_message = message.content
            else:
                anthropic_messages.append(
                    {
                        "role": message.role,
                        "content": message.content or "",
                    }
                )

        return system_message, anthropic_messages

    def _extract_content(self, response: Any) -> str:
        """Extract text content from Anthropic response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request.

        For task-oriented operations. Expects input with 'messages' key
        or a 'prompt' key for simple text generation.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        # Check if input contains messages
        if "messages" in request.input:
            messages_data = request.input["messages"]
            # Convert to Message objects for processing
            from reminix_runtime.types import Message

            messages = [Message(**m) if isinstance(m, dict) else m for m in messages_data]
        elif "prompt" in request.input:
            from reminix_runtime.types import Message

            messages = [Message(role="user", content=request.input["prompt"])]
        else:
            from reminix_runtime.types import Message

            messages = [Message(role="user", content=str(request.input))]

        # Extract system message and convert messages
        system_message, anthropic_messages = self._extract_system_and_messages(messages)

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_message:
            kwargs["system"] = system_message

        # Call Anthropic API
        response = await self._client.messages.create(**kwargs)

        # Extract content from response
        output = self._extract_content(response)

        return InvokeResponse(output=output)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        For conversational interactions.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with output and messages.
        """
        # Extract system message and convert messages
        system_message, anthropic_messages = self._extract_system_and_messages(request.messages)

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_message:
            kwargs["system"] = system_message

        # Call Anthropic API
        response = await self._client.messages.create(**kwargs)

        # Extract content from response
        output = self._extract_content(response)

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
        # Build messages from input
        if "messages" in request.input:
            messages_data = request.input["messages"]
            from reminix_runtime.types import Message

            messages = [Message(**m) if isinstance(m, dict) else m for m in messages_data]
        elif "prompt" in request.input:
            from reminix_runtime.types import Message

            messages = [Message(role="user", content=request.input["prompt"])]
        else:
            from reminix_runtime.types import Message

            messages = [Message(role="user", content=str(request.input))]

        # Extract system message and convert messages
        system_message, anthropic_messages = self._extract_system_and_messages(messages)

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_message:
            kwargs["system"] = system_message

        # Stream from Anthropic API
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield json.dumps({"chunk": text})

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request.

        Args:
            request: The chat request with messages.

        Yields:
            JSON-encoded chunks from the stream.
        """
        # Extract system message and convert messages
        system_message, anthropic_messages = self._extract_system_and_messages(request.messages)

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_message:
            kwargs["system"] = system_message

        # Stream from Anthropic API
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield json.dumps({"chunk": text})


def wrap(
    client: AsyncAnthropic,
    name: str = "anthropic-agent",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
) -> AnthropicAdapter:
    """Wrap an Anthropic client for use with Reminix Runtime.

    Args:
        client: An Anthropic async client.
        name: Name for the agent.
        model: The model to use for completions.
        max_tokens: Maximum tokens in the response.

    Returns:
        An AnthropicAdapter instance.

    Example:
        ```python
        from anthropic import AsyncAnthropic
        from reminix_anthropic import wrap
        from reminix_runtime import serve

        client = AsyncAnthropic()
        agent = wrap(client, name="my-agent", model="claude-sonnet-4-20250514")
        serve([agent], port=8080)
        ```
    """
    return AnthropicAdapter(client, name=name, model=model, max_tokens=max_tokens)
