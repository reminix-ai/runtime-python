"""OpenAI task agent for Reminix Runtime."""

import json
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import AGENT_TYPES, Agent, AgentRequest


class OpenAITaskAgent(Agent):
    """OpenAI task agent using structured outputs (JSON schema)."""

    def __init__(
        self,
        client: AsyncOpenAI,
        output_schema: dict[str, Any],
        *,
        name: str = "openai-task-agent",
        model: str = "gpt-4o-mini",
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "openai task agent",
            streaming=False,
            input_schema=AGENT_TYPES["task"]["input"],
            output_schema=AGENT_TYPES["task"]["output"],
            type="task",
            framework="openai",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._client = client
        self._user_output_schema = output_schema
        self._model = model

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

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        if self.instructions:
            messages.insert(0, {"role": "system", "content": self.instructions})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "task_result",
                    "schema": self._user_output_schema,
                },
            },  # type: ignore
        )

        content = response.choices[0].message.content or "{}"
        output = json.loads(content)
        return {"output": output}
