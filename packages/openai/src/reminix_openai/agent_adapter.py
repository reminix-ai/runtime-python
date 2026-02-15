"""OpenAI agent adapter for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import (
    ADAPTER_INPUT,
    AgentRequest,
    Message,
    build_messages_from_input,
    message_content_to_text,
    serve,
)


class OpenAIAgentAdapter:
    """Agent adapter for OpenAI chat completions."""

    def __init__(
        self,
        client: AsyncOpenAI,
        name: str = "openai-agent",
        model: str = "gpt-4o-mini",
    ) -> None:
        self._client = client
        self._name = name
        self._model = model

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "description": "openai adapter",
            "capabilities": {"streaming": True},
            "input": ADAPTER_INPUT,
            "output": {"type": "string"},
            "adapter": "openai",
        }

    def _to_openai_message(self, message: Message) -> dict[str, Any]:
        """Convert a Reminix message to OpenAI format."""
        role = "system" if message.role == "developer" else message.role
        if role not in ("user", "assistant", "system"):
            role = "user"
        result: dict[str, Any] = {
            "role": role,
            "content": message_content_to_text(message.content) or "",
        }
        if message.tool_calls:
            result["tool_calls"] = message.tool_calls
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        if message.name:
            result["name"] = message.name
        return result

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        messages = build_messages_from_input(request)
        openai_messages = [self._to_openai_message(m) for m in messages]

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
        )

        output = response.choices[0].message.content or ""
        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        messages = build_messages_from_input(request)
        openai_messages = [self._to_openai_message(m) for m in messages]

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
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
    """Wrap an OpenAI client and serve it immediately."""
    agent = wrap_agent(client, name=name, model=model)
    serve(agents=[agent], port=port, host=host)
