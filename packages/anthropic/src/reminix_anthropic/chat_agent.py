"""Anthropic chat agent for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from reminix_runtime import (
    AGENT_TYPES,
    AgentRequest,
    Message,
    build_messages_from_input,
    message_content_to_text,
)


class AnthropicChatAgent:
    """Anthropic chat agent using the messages API."""

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        name: str = "anthropic-agent",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        description: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self._client = client
        self._name = name
        self._model = model
        self._max_tokens = max_tokens
        self._description = description or "anthropic chat agent"
        self._instructions = instructions

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "description": self._description,
            "capabilities": {"streaming": True},
            "input": AGENT_TYPES["chat"]["input"],
            "output": AGENT_TYPES["chat"]["output"],
            "framework": "anthropic",
            "type": "chat",
        }

    def _extract_system_and_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system message and convert remaining messages to Anthropic format."""
        system_message: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            text = message_content_to_text(message.content)
            if message.role == "system" or message.role == "developer":
                system_message = text
            elif message.role in ("user", "assistant"):
                anthropic_messages.append({"role": message.role, "content": text})

        return system_message, anthropic_messages

    def _extract_content(self, response: Any) -> str:
        """Extract text content from Anthropic response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        messages = build_messages_from_input(request)
        system_message, anthropic_messages = self._extract_system_and_messages(messages)
        if self._instructions:
            system_message = self._instructions + (
                "\n\n" + system_message if system_message else ""
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_message:
            kwargs["system"] = system_message

        response = await self._client.messages.create(**kwargs)
        output = self._extract_content(response)
        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        messages = build_messages_from_input(request)
        system_message, anthropic_messages = self._extract_system_and_messages(messages)
        if self._instructions:
            system_message = self._instructions + (
                "\n\n" + system_message if system_message else ""
            )

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": anthropic_messages,
        }
        if system_message:
            kwargs["system"] = system_message

        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield json.dumps({"chunk": text})
