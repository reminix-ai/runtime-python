"""Anthropic task agent for Reminix Runtime."""

import json
from typing import Any

from anthropic import AsyncAnthropic

from reminix_runtime import AGENT_TYPES, Agent, AgentRequest


class AnthropicTaskAgent(Agent):
    """Anthropic task agent using tool-use structured output."""

    def __init__(
        self,
        client: AsyncAnthropic,
        response_schema: dict[str, Any],
        *,
        name: str = "anthropic-task-agent",
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 4096,
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "anthropic task agent",
            streaming=False,
            input_schema=AGENT_TYPES["task"]["input"],
            output_schema=AGENT_TYPES["task"]["output"],
            type="task",
            framework="anthropic",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._client = client
        self._response_schema = response_schema
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

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "tools": [
                {
                    "name": "task_result",
                    "description": "Return the structured result of the task",
                    "input_schema": self._response_schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "task_result"},
        }
        if self.instructions:
            kwargs["system"] = self.instructions

        response = await self._client.messages.create(**kwargs)

        # Extract structured output from tool_use block
        for block in response.content:
            if block.type == "tool_use":
                return {"output": block.input}

        return {"output": {}}
