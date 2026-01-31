"""Anthropic agent adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from reminix_runtime import (
    AgentAdapter,
    AgentInvokeRequest,
    AgentInvokeResponseDict,
    Message,
    serve,
)


class AnthropicAgentAdapter(AgentAdapter):
    """Agent adapter for Anthropic messages API."""

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

    def _build_messages_from_input(self, request: AgentInvokeRequest) -> list[Message]:
        """Build Message list from invoke request input."""
        # Check if input contains messages (chat-style)
        if "messages" in request.input:
            messages_data = request.input["messages"]
            return [Message(**m) if isinstance(m, dict) else m for m in messages_data]
        elif "prompt" in request.input:
            return [Message(role="user", content=request.input["prompt"])]
        else:
            return [Message(role="user", content=str(request.input))]

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponseDict:
        """Handle an invoke request.

        For both task-oriented and chat-style operations. Expects input with 'messages' key
        or a 'prompt' key for simple text generation.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        messages = self._build_messages_from_input(request)

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

        return {"output": output}

    async def invoke_stream(self, request: AgentInvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request.

        Args:
            request: The invoke request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        messages = self._build_messages_from_input(request)

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


def wrap_agent(
    client: AsyncAnthropic,
    name: str = "anthropic-agent",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
) -> AnthropicAgentAdapter:
    """Wrap an Anthropic client for use with Reminix Runtime.

    Args:
        client: An Anthropic async client.
        name: Name for the agent.
        model: The model to use for completions.
        max_tokens: Maximum tokens in the response.

    Returns:
        An AnthropicAgentAdapter instance.

    Example:
        ```python
        from anthropic import AsyncAnthropic
        from reminix_anthropic import wrap_agent
        from reminix_runtime import serve

        client = AsyncAnthropic()
        agent = wrap_agent(client, name="my-agent", model="claude-sonnet-4-20250514")
        serve(agents=[agent], port=8080)
        ```
    """
    return AnthropicAgentAdapter(client, name=name, model=model, max_tokens=max_tokens)


def serve_agent(
    client: AsyncAnthropic,
    name: str = "anthropic-agent",
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 4096,
    port: int = 8080,
    host: str = "0.0.0.0",
) -> None:
    """Wrap an Anthropic client and serve it immediately.

    This is a convenience function that combines `wrap` and `serve` for single-agent setups.

    Args:
        client: An Anthropic async client.
        name: Name for the agent.
        model: The model to use for completions.
        max_tokens: Maximum tokens in the response.
        port: Port to serve on.
        host: Host to bind to.

    Example:
        ```python
        from anthropic import AsyncAnthropic
        from reminix_anthropic import serve_agent

        client = AsyncAnthropic()
        serve_agent(client, name="my-agent", model="claude-sonnet-4-20250514", port=8080)
        ```
    """
    agent = wrap_agent(client, name=name, model=model, max_tokens=max_tokens)
    serve(agents=[agent], port=port, host=host)
