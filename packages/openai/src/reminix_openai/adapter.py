"""OpenAI adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import (
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
    Message,
)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI chat completions."""

    adapter_name = "openai"

    def __init__(
        self,
        client: AsyncOpenAI,
        name: str = "openai-agent",
        model: str = "gpt-4o-mini",
    ) -> None:
        """Initialize the adapter.

        Args:
            client: An OpenAI async client.
            name: Name for the agent.
            model: The model to use for completions.
        """
        self._client = client
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def _to_openai_message(self, message: Message) -> dict[str, Any]:
        """Convert a Reminix message to OpenAI format."""
        result: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.tool_calls:
            result["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        if message.name:
            result["name"] = message.name
        return result

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
            messages = request.input["messages"]
        elif "prompt" in request.input:
            messages = [{"role": "user", "content": request.input["prompt"]}]
        else:
            # Use input as a single user message
            messages = [{"role": "user", "content": str(request.input)}]

        # Call OpenAI API
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
        )

        # Extract content from response
        output = response.choices[0].message.content or ""

        return InvokeResponse(output=output)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        For conversational interactions.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with output and messages.
        """
        # Convert messages to OpenAI format
        openai_messages = [self._to_openai_message(m) for m in request.messages]

        # Call OpenAI API
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
        )

        # Extract content from response
        output = response.choices[0].message.content or ""

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
            messages = request.input["messages"]
        elif "prompt" in request.input:
            messages = [{"role": "user", "content": request.input["prompt"]}]
        else:
            messages = [{"role": "user", "content": str(request.input)}]

        # Stream from OpenAI API
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield json.dumps({"chunk": content})

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request.

        Args:
            request: The chat request with messages.

        Yields:
            JSON-encoded chunks from the stream.
        """
        # Convert messages to OpenAI format
        openai_messages = [self._to_openai_message(m) for m in request.messages]

        # Stream from OpenAI API
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield json.dumps({"chunk": content})


def wrap(
    client: AsyncOpenAI,
    name: str = "openai-agent",
    model: str = "gpt-4o-mini",
) -> OpenAIAdapter:
    """Wrap an OpenAI client for use with Reminix Runtime.

    Args:
        client: An OpenAI async client.
        name: Name for the agent.
        model: The model to use for completions.

    Returns:
        An OpenAIAdapter instance.

    Example:
        ```python
        from openai import AsyncOpenAI
        from reminix_openai import wrap
        from reminix_runtime import serve

        client = AsyncOpenAI()
        agent = wrap(client, name="my-agent", model="gpt-4o")
        serve([agent], port=8080)
        ```
    """
    return OpenAIAdapter(client, name=name, model=model)
