"""Anthropic thread agent for Reminix Runtime."""

import json
from typing import Any

from anthropic import AsyncAnthropic

from reminix_runtime import (
    AGENT_TYPES,
    AgentRequest,
    Message,
    ToolCall,
    ToolLike,
    ToolRequest,
    build_messages_from_input,
    message_content_to_text,
)


class AnthropicThreadAgent:
    """Anthropic thread agent with tool execution loop."""

    def __init__(
        self,
        client: AsyncAnthropic,
        tools: list[ToolLike],
        *,
        name: str = "anthropic-thread-agent",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        max_turns: int = 10,
        description: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self._client = client
        self._tools = {t.name: t for t in tools}
        self._tool_definitions = [self._to_anthropic_tool(t) for t in tools]
        self._name = name
        self._model = model
        self._max_tokens = max_tokens
        self._max_turns = max_turns
        self._description = description or "anthropic thread agent"
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
            "capabilities": {"streaming": False},
            "input": AGENT_TYPES["thread"]["input"],
            "output": AGENT_TYPES["thread"]["output"],
            "framework": "anthropic",
            "type": "thread",
        }

    @staticmethod
    def _to_anthropic_tool(tool: ToolLike) -> dict[str, Any]:
        """Convert a ToolLike to Anthropic tool definition."""
        return {
            "name": tool.name,
            "description": tool.metadata.get("description", ""),
            "input_schema": tool.metadata.get("input", {}),
        }

    def _extract_system_and_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system message and convert remaining messages to Anthropic format."""
        system_message: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for message in messages:
            if message.role == "system" or message.role == "developer":
                system_message = message_content_to_text(message.content)
            elif message.role in ("user", "assistant"):
                text = message_content_to_text(message.content)
                anthropic_messages.append({"role": message.role, "content": text})

        return system_message, anthropic_messages

    def _response_to_message(self, response: Any) -> Message:
        """Convert an Anthropic response to a Reminix Message."""
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        type="function",
                        function={
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    )
                )

        return Message(
            role="assistant",
            content=" ".join(text_parts) if text_parts else "",
            tool_calls=tool_calls if tool_calls else None,
        )

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        messages = build_messages_from_input(request)
        system_message, anthropic_messages = self._extract_system_and_messages(messages)
        if self._instructions:
            system_message = self._instructions + (
                "\n\n" + system_message if system_message else ""
            )

        for _ in range(self._max_turns):
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": self._max_tokens,
                "messages": anthropic_messages,
                "tools": self._tool_definitions,  # type: ignore
            }
            if system_message:
                kwargs["system"] = system_message

            response = await self._client.messages.create(**kwargs)

            # Convert response to Reminix message and add to output
            assistant_msg = self._response_to_message(response)
            messages.append(assistant_msg)

            # Append assistant response content to Anthropic messages
            anthropic_messages.append({"role": "assistant", "content": response.content})

            # Check if there are tool_use blocks
            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            # Execute each tool call and collect results
            tool_results: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                try:
                    tool = self._tools[block.name]
                    result = await tool.call(ToolRequest(input=block.input))
                    tool_result = json.dumps(result.get("output", result))
                except Exception as e:
                    tool_result = f"Error: {e!s}"

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_result,
                    }
                )

                # Add tool result to output messages
                messages.append(
                    Message(
                        role="tool",
                        content=tool_result,
                        tool_call_id=block.id,
                    )
                )

            # Add tool results as a user message for Anthropic
            anthropic_messages.append({"role": "user", "content": tool_results})

        return {"output": [m.model_dump(exclude_none=True) for m in messages]}
