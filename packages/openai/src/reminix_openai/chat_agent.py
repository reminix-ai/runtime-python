"""OpenAI chat agent for Reminix Runtime."""

from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
    AgentRequest,
    build_messages_from_input,
)

from .message_utils import to_openai_message


class OpenAIChatAgent(Agent):
    """OpenAI chat agent using chat completions."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        name: str = "openai-agent",
        model: str = "gpt-4o-mini",
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "openai chat agent",
            streaming=True,
            input_schema=AGENT_TYPES["chat"]["inputSchema"],
            output_schema=AGENT_TYPES["chat"]["outputSchema"],
            type="chat",
            framework="openai",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._client = client
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        messages = build_messages_from_input(request)
        openai_messages = [to_openai_message(m) for m in messages]
        if self.instructions:
            openai_messages.insert(0, {"role": "system", "content": self.instructions})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
        )

        output = response.choices[0].message.content or ""
        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        messages = build_messages_from_input(request)
        openai_messages = [to_openai_message(m) for m in messages]
        if self.instructions:
            openai_messages.insert(0, {"role": "system", "content": self.instructions})

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                yield content
