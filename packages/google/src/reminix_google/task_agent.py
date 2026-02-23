"""Google Gemini task agent for Reminix Runtime."""

import json
from typing import Any, cast

from google import genai
from google.genai import types

from reminix_runtime import AGENT_TYPES, Agent, AgentRequest


class GoogleTaskAgent(Agent):
    """Google Gemini task agent using function-calling structured output."""

    def __init__(
        self,
        client: genai.Client,
        *,
        output_schema: dict[str, Any],
        name: str = "google-task-agent",
        model: str = "gemini-2.5-flash",
        max_tokens: int = 4096,
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "google task agent",
            streaming=False,
            input_schema=AGENT_TYPES["task"]["inputSchema"],
            output_schema=AGENT_TYPES["task"]["outputSchema"],
            type="task",
            framework="google",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._client = client
        self._user_output_schema = output_schema
        self._model = model
        self._max_tokens = max_tokens

    @property
    def model(self) -> str:
        return self._model

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        task = request.input["task"]

        # Include any additional context from input
        extra = {k: v for k, v in request.input.items() if k != "task"}
        prompt = task
        if extra:
            prompt += f"\n\nContext:\n{json.dumps(extra, indent=2)}"

        config = types.GenerateContentConfig(
            max_output_tokens=self._max_tokens,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(
                            name="task_result",
                            description="Return the structured result of the task",
                            parameters=cast(types.Schema, self._user_output_schema),
                        )
                    ],
                )
            ],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowed_function_names=["task_result"],
                ),
            ),
        )
        if self.instructions:
            config.system_instruction = self.instructions

        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=config,
        )

        # Extract structured output from function call
        assert response.candidates
        candidate = response.candidates[0]
        assert candidate.content and candidate.content.parts
        for part in candidate.content.parts:
            if part.function_call and part.function_call.args:
                return {"output": dict(part.function_call.args)}

        return {"output": {}}
