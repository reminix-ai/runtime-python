"""OpenAI agent adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import (
    AgentAdapter,
    AgentInvokeRequest,
    AgentInvokeResponseDict,
    Message,
    serve,
)


class OpenAIAgentAdapter(AgentAdapter):
    """Agent adapter for OpenAI chat completions."""

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

    def _build_openai_messages(self, request: AgentInvokeRequest) -> list[dict[str, Any]]:
        """Build OpenAI messages from invoke request input."""
        # Check if input contains messages (chat-style)
        if "messages" in request.input:
            messages_data = request.input["messages"]
            # Convert to Message objects if needed, then to OpenAI format
            messages = [Message(**m) if isinstance(m, dict) else m for m in messages_data]
            return [self._to_openai_message(m) for m in messages]
        elif "prompt" in request.input:
            return [{"role": "user", "content": request.input["prompt"]}]
        else:
            # Use input as a single user message
            return [{"role": "user", "content": str(request.input)}]

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponseDict:
        """Handle an invoke request.

        For both task-oriented and chat-style operations. Expects input with 'messages' key
        or a 'prompt' key for simple text generation.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        messages = self._build_openai_messages(request)

        # Call OpenAI API
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
        )

        # Extract content from response
        output = response.choices[0].message.content or ""

        return {"output": output}

    async def invoke_stream(self, request: AgentInvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request.

        Args:
            request: The invoke request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        messages = self._build_openai_messages(request)

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


def wrap_agent(
    client: AsyncOpenAI,
    name: str = "openai-agent",
    model: str = "gpt-4o-mini",
) -> OpenAIAgentAdapter:
    """Wrap an OpenAI client for use with Reminix Runtime.

    Args:
        client: An OpenAI async client.
        name: Name for the agent.
        model: The model to use for completions.

    Returns:
        An OpenAIAgentAdapter instance.

    Example:
        ```python
        from openai import AsyncOpenAI
        from reminix_openai import wrap_agent
        from reminix_runtime import serve

        client = AsyncOpenAI()
        agent = wrap_agent(client, name="my-agent", model="gpt-4o")
        serve(agents=[agent], port=8080)
        ```
    """
    return OpenAIAgentAdapter(client, name=name, model=model)


def serve_agent(
    client: AsyncOpenAI,
    name: str = "openai-agent",
    model: str = "gpt-4o-mini",
    port: int = 8080,
    host: str = "0.0.0.0",
) -> None:
    """Wrap an OpenAI client and serve it immediately.

    This is a convenience function that combines `wrap` and `serve` for single-agent setups.

    Args:
        client: An OpenAI async client.
        name: Name for the agent.
        model: The model to use for completions.
        port: Port to serve on.
        host: Host to bind to.

    Example:
        ```python
        from openai import AsyncOpenAI
        from reminix_openai import serve_agent

        client = AsyncOpenAI()
        serve_agent(client, name="my-agent", model="gpt-4o", port=8080)
        ```
    """
    agent = wrap_agent(client, name=name, model=model)
    serve(agents=[agent], port=port, host=host)
