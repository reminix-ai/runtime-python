"""OpenAI task agent for Reminix Runtime."""

import json
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import AGENT_TEMPLATES, AgentRequest


class OpenAITaskAgent:
    """OpenAI task agent using structured outputs (JSON schema)."""

    def __init__(
        self,
        client: AsyncOpenAI,
        output_schema: dict[str, Any],
        name: str = "openai-task-agent",
        model: str = "gpt-4o-mini",
    ) -> None:
        self._client = client
        self._output_schema = output_schema
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
            "description": "openai task agent",
            "capabilities": {"streaming": False},
            "input": AGENT_TEMPLATES["task"]["input"],
            "output": AGENT_TEMPLATES["task"]["output"],
            "adapter": "openai",
            "template": "task",
        }

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        task = request.input["task"]

        # Include any additional context from input
        extra = {k: v for k, v in request.input.items() if k != "task"}
        prompt = task
        if extra:
            prompt += f"\n\nContext:\n{json.dumps(extra, indent=2)}"

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "task_result",
                    "schema": self._output_schema,
                },
            },  # type: ignore
        )

        content = response.choices[0].message.content or "{}"
        output = json.loads(content)
        return {"output": output}
