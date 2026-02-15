"""Anthropic task agent for Reminix Runtime."""

import json
from typing import Any

from anthropic import AsyncAnthropic

from reminix_runtime import AGENT_TYPES, AgentRequest


class AnthropicTaskAgent:
    """Anthropic task agent using tool-use structured output."""

    def __init__(
        self,
        client: AsyncAnthropic,
        output_schema: dict[str, Any],
        *,
        name: str = "anthropic-task-agent",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        description: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self._client = client
        self._output_schema = output_schema
        self._name = name
        self._model = model
        self._max_tokens = max_tokens
        self._description = description or "anthropic task agent"
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
            "input": AGENT_TYPES["task"]["input"],
            "output": AGENT_TYPES["task"]["output"],
            "framework": "anthropic",
            "type": "task",
        }

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        task = request.input["task"]

        # Include any additional context from input
        extra = {k: v for k, v in request.input.items() if k != "task"}
        prompt = task
        if extra:
            prompt += f"\n\nContext:\n{json.dumps(extra, indent=2)}"

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {
                    "name": "task_result",
                    "description": "Return the structured result of the task",
                    "input_schema": self._output_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "task_result"},
        }
        if self._instructions:
            kwargs["system"] = self._instructions

        response = await self._client.messages.create(**kwargs)

        # Extract structured output from tool_use block
        for block in response.content:
            if block.type == "tool_use":
                return {"output": block.input}

        return {"output": {}}
