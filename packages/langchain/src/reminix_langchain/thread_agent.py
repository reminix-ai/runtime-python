"""LangChain thread agent for Reminix Runtime.

Accepts a CompiledStateGraph (from create_agent, with tools) or a plain Runnable.
Returns the full message thread including tool calls and results.
"""

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import Runnable

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
    AgentRequest,
    build_messages_from_input,
)

from .message_utils import from_langchain_message, to_langchain_message


def _is_compiled_state_graph(agent: Runnable) -> bool:
    """Detect if a runnable is a CompiledStateGraph (from langgraph create_agent)."""
    return hasattr(agent, "get_graph") and callable(agent.get_graph)


class LangChainThreadAgent(Agent):
    """LangChain thread agent for graphs and runnables."""

    def __init__(
        self,
        agent: Runnable,
        *,
        name: str = "langchain-thread-agent",
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "langchain thread agent",
            streaming=False,
            input_schema=AGENT_TYPES["thread"]["inputSchema"],
            output_schema=AGENT_TYPES["thread"]["outputSchema"],
            type="thread",
            framework="langchain",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._agent = agent
        self._is_graph = _is_compiled_state_graph(agent)

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        input_messages = build_messages_from_input(request)
        lc_messages: list[BaseMessage] = [to_langchain_message(m) for m in input_messages]

        if self.instructions:
            lc_messages.insert(0, SystemMessage(content=self.instructions))

        if self._is_graph:
            # CompiledStateGraph: invoke with { messages } dict, result is { messages }
            result = await self._agent.ainvoke({"messages": lc_messages})
            result_messages: list[BaseMessage] = result["messages"]
        else:
            # Plain Runnable: invoke with messages array, result is a single message
            result = await self._agent.ainvoke(lc_messages)
            if isinstance(result, list):
                result_messages = result
            else:
                result_messages = [*lc_messages, result]

        # Convert all result messages to Reminix format
        output = []
        for m in result_messages:
            msg = from_langchain_message(m)
            # Strip None fields
            clean: dict[str, Any] = {"role": msg["role"], "content": msg["content"]}
            if msg.get("tool_calls"):
                clean["tool_calls"] = msg["tool_calls"]
            if msg.get("tool_call_id"):
                clean["tool_call_id"] = msg["tool_call_id"]
            if msg.get("name"):
                clean["name"] = msg["name"]
            output.append(clean)

        return {"output": output}
