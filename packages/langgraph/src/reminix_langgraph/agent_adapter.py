"""LangGraph agent adapter for Reminix Runtime."""

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

from reminix_runtime import (
    AgentAdapter,
    AgentInvokeRequest,
    AgentInvokeResponseDict,
    Message,
    serve,
)


class LangGraphAgentAdapter(AgentAdapter):
    """Agent adapter for LangGraph compiled graphs."""

    adapter_name = "langgraph"

    def __init__(self, graph: Any, name: str = "langgraph-agent") -> None:
        """Initialize the adapter.

        Args:
            graph: A LangGraph compiled graph.
            name: Name for the agent.
        """
        self._graph = graph
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _to_langchain_message(self, message: Message) -> BaseMessage:
        """Convert a Reminix message to a LangChain message."""
        role = message.role
        content = message.content or ""

        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        elif role == "tool":
            tool_call_id = getattr(message, "tool_call_id", None) or "unknown"
            return ToolMessage(content=content, tool_call_id=tool_call_id)
        else:
            return HumanMessage(content=content)

    def _to_reminix_message(self, message: BaseMessage) -> dict[str, Any]:
        """Convert a LangChain message to a Reminix message dict."""
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            role = "assistant"

        content = message.content if isinstance(message.content, str) else str(message.content)
        return {"role": role, "content": content}

    def _get_last_ai_content(self, messages: list[BaseMessage]) -> str:
        """Extract content from the last AI message."""
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message.content if isinstance(message.content, str) else str(message.content)
        return ""

    def _build_graph_input(self, request: AgentInvokeRequest) -> Any:
        """Build LangGraph input from invoke request."""
        # Check if input contains messages (chat-style)
        if "messages" in request.input:
            messages_data = request.input["messages"]
            messages = [Message(**m) if isinstance(m, dict) else m for m in messages_data]
            lc_messages = [self._to_langchain_message(m) for m in messages]
            return {"messages": lc_messages}
        else:
            # Pass input directly to the graph
            return request.input

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponseDict:
        """Handle an invoke request.

        For both task-oriented and chat-style operations.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        graph_input = self._build_graph_input(request)

        # Call the graph
        result = await self._graph.ainvoke(graph_input)

        # Extract output from result
        if isinstance(result, dict) and "messages" in result:
            messages = result.get("messages", [])
            output = self._get_last_ai_content(messages)
        elif isinstance(result, dict):
            output = result
        else:
            output = str(result)

        return {"output": output}

    async def invoke_stream(self, request: AgentInvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request.

        Args:
            request: The invoke request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        graph_input = self._build_graph_input(request)

        async for chunk in self._graph.astream(graph_input):
            # LangGraph streams dicts with node outputs
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


def wrap_agent(graph: Any, name: str = "langgraph-agent") -> LangGraphAgentAdapter:
    """Wrap a LangGraph compiled graph for use with Reminix Runtime.

    Args:
        graph: A LangGraph compiled graph.
        name: Name for the agent.

    Returns:
        A LangGraphAgentAdapter instance.

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from reminix_langgraph import wrap_agent
        from reminix_runtime import serve

        llm = ChatOpenAI(model="gpt-4")
        graph = create_react_agent(llm, tools=[])
        agent = wrap_agent(graph, name="my-agent")
        serve(agents=[agent], port=8080)
        ```
    """
    return LangGraphAgentAdapter(graph, name=name)


def serve_agent(
    graph: Any,
    name: str = "langgraph-agent",
    port: int = 8080,
    host: str = "0.0.0.0",
) -> None:
    """Wrap a LangGraph graph and serve it immediately.

    This is a convenience function that combines `wrap` and `serve` for single-agent setups.

    Args:
        graph: A LangGraph compiled graph.
        name: Name for the agent.
        port: Port to serve on.
        host: Host to bind to.

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from reminix_langgraph import serve_agent

        llm = ChatOpenAI(model="gpt-4")
        graph = create_react_agent(llm, tools=[])
        serve_agent(graph, name="my-agent", port=8080)
        ```
    """
    agent = wrap_agent(graph, name=name)
    serve(agents=[agent], port=port, host=host)
