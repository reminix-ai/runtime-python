"""LangChain chat agent for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

from reminix_runtime import (
    AGENT_TYPES,
    Agent,
    AgentRequest,
    Message,
    build_messages_from_input,
    message_content_to_text,
)


def to_langchain_message(message: Message) -> BaseMessage:
    """Convert a Reminix message to a LangChain message.

    This is exported for reuse by the langgraph package.
    """
    role = message.role
    content = message_content_to_text(message.content)

    if role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    elif role == "system" or role == "developer":
        return SystemMessage(content=content)
    elif role == "tool":
        tool_call_id = getattr(message, "tool_call_id", None) or "unknown"
        return ToolMessage(content=content, tool_call_id=tool_call_id)
    else:
        return HumanMessage(content=content)


class LangChainChatAgent(Agent):
    """LangChain chat agent for agents and runnables."""

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
            input_schema=AGENT_TYPES["chat"]["input"],
            output_schema=AGENT_TYPES["chat"]["output"],
            type="chat",
            framework="langchain",
            instructions=instructions,
            tags=tags,
            metadata=metadata,
        )
        self._agent = agent

    def _build_langchain_input(self, request: AgentRequest) -> Any:
        """Build LangChain input from invoke request."""
        messages = build_messages_from_input(request)

        # If input had messages, convert to LangChain format
        if "messages" in request.input:
            lc_messages = [to_langchain_message(m) for m in messages]
            if self.instructions:
                lc_messages.insert(0, SystemMessage(content=self.instructions))
            return lc_messages
        elif "prompt" in request.input:
            return request.input["prompt"]
        else:
            return request.input

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        invoke_input = self._build_langchain_input(request)
        response = await self._agent.ainvoke(invoke_input)

        if isinstance(response, BaseMessage):
            output = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        elif isinstance(response, dict):
            output = response
        else:
            output = str(response)

        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        stream_input = self._build_langchain_input(request)

        async for chunk in self._agent.astream(stream_input):
            if isinstance(chunk, (BaseMessage, AIMessageChunk)):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            elif isinstance(chunk, dict):
                content = json.dumps(chunk)
            else:
                content = str(chunk)
            yield content
