"""LangGraph thread agent for Reminix Runtime."""

import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage

from reminix_langchain import to_langchain_message
from reminix_runtime import (
    AGENT_TEMPLATES,
    AgentRequest,
    build_messages_from_input,
)


class LangGraphThreadAgent:
    """LangGraph thread agent for compiled graphs."""

    def __init__(self, graph: Any, name: str = "langgraph-agent") -> None:
        self._graph = graph
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "description": "langgraph thread agent",
            "capabilities": {"streaming": True},
            "input": AGENT_TEMPLATES["thread"]["input"],
            "output": {"type": "string"},
            "framework": "langgraph",
            "template": "thread",
        }

    def _get_last_ai_content(self, messages: list[BaseMessage]) -> str:
        """Extract content from the last AI message."""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message.content if isinstance(message.content, str) else str(message.content)
        return ""

    def _build_graph_input(self, request: AgentRequest) -> Any:
        """Build LangGraph input from invoke request."""
        if "messages" in request.input:
            messages = build_messages_from_input(request)
            lc_messages = [to_langchain_message(m) for m in messages]
            return {"messages": lc_messages}
        else:
            return request.input

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        graph_input = self._build_graph_input(request)
        result = await self._graph.ainvoke(graph_input)

        if isinstance(result, dict) and "messages" in result:
            messages = result.get("messages", [])
            output = self._get_last_ai_content(messages)
        elif isinstance(result, dict):
            output = result
        else:
            output = str(result)

        return {"output": output}

    async def invoke_stream(self, request: AgentRequest) -> AsyncIterator[str]:
        graph_input = self._build_graph_input(request)

        async for chunk in self._graph.astream(graph_input):
            if isinstance(chunk, dict):
                for _node_name, node_output in chunk.items():
                    if isinstance(node_output, dict) and "messages" in node_output:
                        for msg in node_output["messages"]:
                            if isinstance(msg, (AIMessage, AIMessageChunk)):
                                content = (
                                    msg.content
                                    if isinstance(msg.content, str)
                                    else str(msg.content)
                                )
                                if content:
                                    yield json.dumps({"chunk": content})
                    else:
                        yield json.dumps({"chunk": json.dumps(node_output)})
            else:
                yield json.dumps({"chunk": str(chunk)})
