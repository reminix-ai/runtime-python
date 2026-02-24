"""LangChain chat agent for Reminix Runtime.

Accepts a CompiledStateGraph (from create_agent, with tools) or a plain Runnable.
"""

import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
)
from langchain_core.runnables import Runnable

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
    AgentRequest,
    build_messages_from_input,
)

from .message_utils import to_langchain_message


def _is_compiled_state_graph(agent: Runnable) -> bool:
    """Detect if a runnable is a CompiledStateGraph (from langgraph create_agent)."""
    return hasattr(agent, "get_graph") and callable(agent.get_graph)


class LangChainChatAgent(Agent):
    """LangChain chat agent for graphs and runnables."""

    def __init__(
        self,
        agent: Runnable,
        *,
        name: str = "langchain-agent",
        description: str | None = None,
        instructions: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            name,
            description=description or "langchain chat agent",
            streaming=True,
            input_schema=AGENT_TYPES["chat"]["inputSchema"],
            output_schema=AGENT_TYPES["chat"]["outputSchema"],
            type="chat",
            framework="langchain",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._agent = agent
        self._is_graph = _is_compiled_state_graph(agent)

    def _build_langchain_input(self, request: AgentRequest) -> Any:
        """Build LangChain input from invoke request."""
        messages = build_messages_from_input(request)

        if "messages" in request.input:
            lc_messages = [to_langchain_message(m) for m in messages]
            if self.instructions:
                lc_messages.insert(0, SystemMessage(content=self.instructions))
            if self._is_graph:
                return {"messages": lc_messages}
            return lc_messages
        elif "prompt" in request.input:
            prompt = request.input["prompt"]
            if self._is_graph:
                return {"messages": [SystemMessage(content=str(prompt))]}
            return prompt
        else:
            if self._is_graph:
                return {"messages": [SystemMessage(content=json.dumps(request.input))]}
            return request.input

    def _extract_output(self, response: Any) -> Any:
        """Extract text output from response."""
        if self._is_graph:
            # CompiledStateGraph returns { messages: BaseMessage[] }
            messages: list[BaseMessage] = response["messages"]
            if messages:
                last_msg = messages[-1]
                content = last_msg.content
                return content if isinstance(content, str) else str(content)
            return ""

        if isinstance(response, BaseMessage):
            return response.content if isinstance(response.content, str) else str(response.content)
        elif isinstance(response, dict):
            return response
        else:
            return str(response)

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        invoke_input = self._build_langchain_input(request)
        response = await self._agent.ainvoke(invoke_input)
        return {"output": self._extract_output(response)}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        stream_input = self._build_langchain_input(request)

        async for chunk in self._agent.astream(stream_input):
            if self._is_graph:
                # Graph streams events — extract text content from message chunks
                if isinstance(chunk, dict) and "messages" in chunk:
                    messages = chunk["messages"]
                    if messages:
                        last_msg = messages[-1]
                        content = (
                            last_msg.content
                            if isinstance(last_msg.content, str)
                            else str(last_msg.content)
                        )
                        yield content
                continue

            if isinstance(chunk, (BaseMessage, AIMessageChunk)):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            elif isinstance(chunk, dict):
                content = json.dumps(chunk)
            else:
                content = str(chunk)
            yield content
