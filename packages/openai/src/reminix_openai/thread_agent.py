"""OpenAI thread agent for Reminix Runtime."""

import json
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
    AgentRequest,
    Message,
    Tool,
    ToolCall,
    ToolRequest,
    build_messages_from_input,
    message_content_to_text,
)


class OpenAIThreadAgent(Agent):
    """OpenAI thread agent with tool execution loop."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        tools: list[Tool],
        name: str = "openai-thread-agent",
        model: str = "gpt-4o-mini",
        max_turns: int = 10,
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "openai thread agent",
            streaming=False,
            input_schema=AGENT_TYPES["thread"]["input"],
            output_schema=AGENT_TYPES["thread"]["output"],
            type="thread",
            framework="openai",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._client = client
        self._tools = {t.name: t for t in tools}
        self._tool_definitions = [self._to_openai_tool(t) for t in tools]
        self._model = model
        self._max_turns = max_turns

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _to_openai_tool(tool: Tool) -> dict[str, Any]:
        """Convert a Tool to OpenAI tool definition."""
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.metadata.get("description", ""),
                "parameters": tool.metadata.get("inputSchema", {}),
            },
        }

    def _to_openai_message(self, message: Message) -> dict[str, Any]:
        """Convert a Reminix message to OpenAI format."""
        role = "system" if message.role == "developer" else message.role
        if role not in ("user", "assistant", "system", "tool"):
            role = "user"
        result: dict[str, Any] = {
            "role": role,
            "content": message_content_to_text(message.content) or "",
        }
        if message.tool_calls:
            result["tool_calls"] = [tc.model_dump() for tc in message.tool_calls]
        if message.tool_call_id:
            result["tool_call_id"] = message.tool_call_id
        if message.name:
            result["name"] = message.name
        return result

    def _response_to_message(self, response_message: Any) -> Message:
        """Convert an OpenAI response message to a Reminix Message."""
        tool_calls = None
        if response_message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    type="function",
                    function={"name": tc.function.name, "arguments": tc.function.arguments},
                )
                for tc in response_message.tool_calls
            ]
        return Message(
            role="assistant",
            content=response_message.content or "",
            tool_calls=tool_calls,
        )

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        messages = build_messages_from_input(request)
        openai_messages = [self._to_openai_message(m) for m in messages]
        if self.instructions:
            openai_messages.insert(0, {"role": "system", "content": self.instructions})

        for _ in range(self._max_turns):
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,  # type: ignore
                tools=self._tool_definitions,  # type: ignore
            )

            response_message = response.choices[0].message

            # Append assistant message (with tool_calls intact)
            assistant_dict: dict[str, Any] = {
                "role": "assistant",
                "content": response_message.content or "",
            }
            # Filter to function tool calls only
            fn_tool_calls: list[Any] = [
                tc for tc in (response_message.tool_calls or []) if tc.type == "function"
            ]
            if fn_tool_calls:
                assistant_dict["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in fn_tool_calls
                ]
            openai_messages.append(assistant_dict)

            # Add assistant message to output
            assistant_msg = self._response_to_message(response_message)
            messages.append(assistant_msg)

            # If no tool calls, we're done
            if not fn_tool_calls:
                break

            # Execute each tool call
            for tc in fn_tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                    tool = self._tools[tool_name]
                    result = await tool.call(ToolRequest(arguments=args))
                    tool_result = json.dumps(result.get("output", result))
                except Exception as e:
                    tool_result = f"Error: {e!s}"

                # Append tool result to OpenAI messages
                openai_messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_result}
                )
                # Append tool result to output messages
                messages.append(Message(role="tool", content=tool_result, tool_call_id=tc.id))

        return {"output": [m.model_dump(exclude_none=True) for m in messages]}
