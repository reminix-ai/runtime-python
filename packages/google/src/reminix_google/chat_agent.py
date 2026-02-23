"""Google Gemini chat agent for Reminix Runtime."""

from collections.abc import AsyncIterator
from typing import Any, cast

from google import genai
from google.genai import types

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
    AgentRequest,
    Message,
    build_messages_from_input,
    message_content_to_text,
)


class GoogleChatAgent(Agent):
    """Google Gemini chat agent using the generate content API."""

    def __init__(
        self,
        client: genai.Client,
        *,
        name: str = "google-agent",
        model: str = "gemini-2.5-flash",
        max_tokens: int = 4096,
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "google chat agent",
            streaming=True,
            input_schema=AGENT_TYPES["chat"]["inputSchema"],
            output_schema=AGENT_TYPES["chat"]["outputSchema"],
            type="chat",
            framework="google",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    def _extract_system_and_contents(
        self, messages: list[Message]
    ) -> tuple[str | None, list[types.Content]]:
        """Extract system message and convert remaining messages to Gemini format."""
        system_message: str | None = None
        contents: list[types.Content] = []

        for message in messages:
            text = message_content_to_text(message.content)
            if message.role == "system" or message.role == "developer":
                system_message = text
            elif message.role == "user":
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=text)]))
            elif message.role == "assistant":
                contents.append(
                    types.Content(role="model", parts=[types.Part.from_text(text=text)])
                )

        return system_message, contents

    def _extract_text(self, response: types.GenerateContentResponse) -> str:
        """Extract text content from Gemini response."""
        return response.text or ""

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        messages = build_messages_from_input(request)
        system_message, contents = self._extract_system_and_contents(messages)
        if self.instructions:
            system_message = self.instructions + ("\n\n" + system_message if system_message else "")

        config = types.GenerateContentConfig(max_output_tokens=self._max_tokens)
        if system_message:
            config.system_instruction = system_message

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=cast(types.ContentListUnion, contents),
            config=config,
        )

        output = self._extract_text(response)
        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        messages = build_messages_from_input(request)
        system_message, contents = self._extract_system_and_contents(messages)
        if self.instructions:
            system_message = self.instructions + ("\n\n" + system_message if system_message else "")

        config = types.GenerateContentConfig(max_output_tokens=self._max_tokens)
        if system_message:
            config.system_instruction = system_message

        stream = await self._client.aio.models.generate_content_stream(
            model=self._model,
            contents=cast(types.ContentListUnion, contents),
            config=config,
        )
        async for chunk in cast(AsyncIterator[types.GenerateContentResponse], stream):
            if chunk.text:
                yield chunk.text
