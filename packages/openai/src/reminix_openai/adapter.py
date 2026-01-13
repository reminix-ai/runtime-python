"""OpenAI adapter for Reminix Runtime."""

from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import (
    BaseAdapter,
    InvokeRequest,
    InvokeResponse,
    ChatRequest,
    ChatResponse,
    Message,
)


class OpenAIAdapter(BaseAdapter):
    """Adapter for OpenAI chat completions."""

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

    def _to_openai_message(self, message: Message) -> dict[str, str]:
        """Convert a Reminix message to OpenAI format."""
        return {"role": message.role, "content": message.content}

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request.

        Args:
            request: The invoke request with messages.

        Returns:
            The invoke response with the assistant's reply.
        """
        # Convert messages to OpenAI format
        openai_messages = [self._to_openai_message(m) for m in request.messages]

        # Call OpenAI API
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
        )

        # Extract content from response
        content = response.choices[0].message.content or ""

        # Build response messages (original + assistant response)
        response_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append({"role": "assistant", "content": content})

        return InvokeResponse(content=content, messages=response_messages)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with the assistant's reply.
        """
        # Convert messages to OpenAI format
        openai_messages = [self._to_openai_message(m) for m in request.messages]

        # Call OpenAI API
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=openai_messages,  # type: ignore
        )

        # Extract content from response
        content = response.choices[0].message.content or ""

        # Build response messages (original + assistant response)
        response_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append({"role": "assistant", "content": content})

        return ChatResponse(content=content, messages=response_messages)


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
