"""OpenAI task agent for Reminix Runtime."""

import json
from typing import Any

from openai import AsyncOpenAI

from reminix_runtime import AGENT_TYPES, AgentRequest


class OpenAITaskAgent:
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
        self._client = client
        self._output_schema = output_schema
        self._name = name
        self._model = model
        self._description = description or "openai task agent"
        self._instructions = instructions
        self._tags = tags
        self._extra_metadata = metadata

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "description": self._description,
            "capabilities": {"streaming": False},
            "input": AGENT_TYPES["task"]["input"],
            "output": AGENT_TYPES["task"]["output"],
            "framework": "openai",
            "type": "task",
        }
        if self._tags:
            result["tags"] = self._tags
        if self._extra_metadata:
            result.update(self._extra_metadata)
        return result

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        task = request.input["task"]

        # Include any additional context from input
        extra = {k: v for k, v in request.input.items() if k != "task"}
        prompt = task
        if extra:
            prompt += f"\n\nContext:\n{json.dumps(extra, indent=2)}"

        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        if self._instructions:
            messages.insert(0, {"role": "system", "content": self._instructions})

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore
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
