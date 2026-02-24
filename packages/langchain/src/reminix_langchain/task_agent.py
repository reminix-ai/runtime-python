"""LangChain task agent for Reminix Runtime.

Accepts a CompiledStateGraph (from create_agent, with tools) or a plain Runnable.
Returns structured output from a single-shot task execution.
"""

import json
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import Runnable

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
    AgentRequest,
)


def _is_compiled_state_graph(agent: Runnable) -> bool:
    """Detect if a runnable is a CompiledStateGraph (from langgraph create_agent)."""
    return hasattr(agent, "get_graph") and callable(agent.get_graph)


class LangChainTaskAgent(Agent):
    """LangChain task agent for graphs and runnables."""

    def __init__(
        self,
        agent: Runnable,
        *,
        name: str = "langchain-task-agent",
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "langchain task agent",
            streaming=False,
            input_schema=AGENT_TYPES["task"]["inputSchema"],
            output_schema=AGENT_TYPES["task"]["outputSchema"],
            type="task",
            framework="langchain",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._agent = agent
        self._is_graph = _is_compiled_state_graph(agent)

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        task = request.input.get("task")
        prompt = task if isinstance(task, str) else json.dumps(request.input)

        if self._is_graph:
            # CompiledStateGraph: invoke with { messages } containing the task prompt
            result = await self._agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            messages: list[BaseMessage] = result["messages"]
            last_message = messages[-1] if messages else None
            if last_message is not None:
                content = last_message.content
                if isinstance(content, str):
                    try:
                        output: Any = json.loads(content)
                    except (json.JSONDecodeError, ValueError):
                        output = content
                else:
                    output = content
            else:
                output = None
        else:
            # Plain Runnable: invoke directly with the prompt
            result = await self._agent.ainvoke(prompt)
            if isinstance(result, BaseMessage):
                content = result.content
                if isinstance(content, str):
                    try:
                        output = json.loads(content)
                    except (json.JSONDecodeError, ValueError):
                        output = content
                else:
                    output = content
            elif isinstance(result, dict):
                output = result
            else:
                output = str(result)

        return {"output": output}
