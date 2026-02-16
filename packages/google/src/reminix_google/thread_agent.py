"""Google Gemini thread agent for Reminix Runtime."""

import json
from typing import Any, cast

from google import genai
from google.genai import types

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


class GoogleThreadAgent(Agent):
    """Google Gemini thread agent with tool execution loop."""

    def __init__(
        self,
        client: genai.Client,
        *,
        tools: list[Tool],
        name: str = "google-thread-agent",
        model: str = "gemini-2.5-flash",
        max_tokens: int = 4096,
        max_turns: int = 10,
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "google thread agent",
            streaming=False,
            input_schema=AGENT_TYPES["thread"]["input"],
            output_schema=AGENT_TYPES["thread"]["output"],
            type="thread",
            framework="google",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._client = client
        self._tools = {t.name: t for t in tools}
        self._function_declarations: list[types.FunctionDeclaration] = [
            self._to_function_declaration(t) for t in tools
        ]
        self._model = model
        self._max_tokens = max_tokens
        self._max_turns = max_turns

    @property
    def model(self) -> str:
        return self._model

    @staticmethod
    def _to_function_declaration(tool: Tool) -> types.FunctionDeclaration:
        """Convert a Tool to a Gemini function declaration."""
        return types.FunctionDeclaration(
            name=tool.name,
            description=tool.metadata.get("description", ""),
            parameters=cast(types.Schema, tool.metadata.get("input", {})),
        )

    def _extract_system_and_contents(self, messages: list[Message]) -> tuple[str | None, list[Any]]:
        """Extract system message and convert remaining messages to Gemini format."""
        system_message: str | None = None
        contents: list[Any] = []

        for message in messages:
            if message.role == "system" or message.role == "developer":
                system_message = message_content_to_text(message.content)
            elif message.role == "user":
                text = message_content_to_text(message.content)
                contents.append({"role": "user", "parts": [{"text": text}]})
            elif message.role == "assistant":
                text = message_content_to_text(message.content)
                contents.append({"role": "model", "parts": [{"text": text}]})

        return system_message, contents

    def _response_to_message(self, response: types.GenerateContentResponse) -> Message:
        """Convert a Gemini response to a Reminix Message."""
        assert response.candidates
        candidate = response.candidates[0]
        assert candidate.content and candidate.content.parts
        parts = candidate.content.parts
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for part in parts:
            if part.text:
                text_parts.append(part.text)
            elif part.function_call:
                fc = part.function_call
                tool_calls.append(
                    ToolCall(
                        id=f"call_{fc.name}_{id(part)}",
                        type="function",
                        function={
                            "name": fc.name or "",
                            "arguments": json.dumps(dict(fc.args)) if fc.args else "{}",
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
        system_message, contents = self._extract_system_and_contents(messages)
        if self.instructions:
            system_message = self.instructions + ("\n\n" + system_message if system_message else "")

        for _ in range(self._max_turns):
            config = types.GenerateContentConfig(
                max_output_tokens=self._max_tokens,
                tools=[types.Tool(function_declarations=self._function_declarations)],
            )
            if system_message:
                config.system_instruction = system_message

            response = await self._client.aio.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )

            # Convert response to Reminix message and add to output
            assistant_msg = self._response_to_message(response)
            messages.append(assistant_msg)

            # Append model response content to Gemini contents
            assert response.candidates
            candidate = response.candidates[0]
            assert candidate.content
            contents.append(candidate.content)

            # Check for function calls
            assert candidate.content.parts
            function_call_parts = [p for p in candidate.content.parts if p.function_call]
            if not function_call_parts:
                break

            # Execute each tool call and collect function responses
            function_response_parts: list[dict[str, Any]] = []
            for part in function_call_parts:
                assert part.function_call
                fc_name = part.function_call.name or ""
                fc_args = dict(part.function_call.args) if part.function_call.args else {}
                try:
                    tool = self._tools[fc_name]
                    result = await tool.call(ToolRequest(input=fc_args))
                    tool_result = result.get("output", result)
                except Exception as e:
                    tool_result = {"error": str(e)}

                function_response_parts.append(
                    {
                        "function_response": {
                            "name": fc_name,
                            "response": tool_result,
                        }
                    }
                )

                # Add tool result to output messages
                call_id = fc_name
                if assistant_msg.tool_calls:
                    for tc in assistant_msg.tool_calls:
                        if tc.function["name"] == fc_name:
                            call_id = tc.id
                            break

                messages.append(
                    Message(
                        role="tool",
                        content=json.dumps(tool_result),
                        tool_call_id=call_id,
                    )
                )

            # Add function responses as user turn for Gemini
            contents.append({"role": "user", "parts": function_response_parts})

        return {"output": [m.model_dump(exclude_none=True) for m in messages]}
