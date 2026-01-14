"""LangGraph adapter for Reminix Runtime."""

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
    BaseAdapter,
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
    Message,
)


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangGraph compiled graphs."""

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

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request.

        For task-oriented operations. Passes the input directly to the graph.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        # Pass input directly to the graph
        result = await self._graph.ainvoke(request.input)

        # Extract output from result
        if isinstance(result, dict) and "messages" in result:
            messages = result.get("messages", [])
            output = self._get_last_ai_content(messages)
        elif isinstance(result, dict):
            output = result
        else:
            output = str(result)

        return InvokeResponse(output=output)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        For conversational interactions. Converts messages to LangChain format
        and invokes the graph with the state dict format.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with output and messages.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in request.messages]

        # Call the graph with state dict format
        result = await self._graph.ainvoke({"messages": lc_messages})

        # Extract messages from result
        result_messages: list[BaseMessage] = result.get("messages", [])

        # Get content from the last AI message
        output = self._get_last_ai_content(result_messages)

        # Convert all messages back to Reminix format
        response_messages = [self._to_reminix_message(m) for m in result_messages]

        return ChatResponse(output=output, messages=response_messages)

    async def invoke_stream(self, request: InvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request.

        Args:
            request: The invoke request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        async for chunk in self._graph.astream(request.input):
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

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request.

        Args:
            request: The chat request with messages.

        Yields:
            JSON-encoded chunks from the stream.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in request.messages]

        # Stream from the graph
        async for chunk in self._graph.astream({"messages": lc_messages}):
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


def wrap(graph: Any, name: str = "langgraph-agent") -> LangGraphAdapter:
    """Wrap a LangGraph compiled graph for use with Reminix Runtime.

    Args:
        graph: A LangGraph compiled graph.
        name: Name for the agent.

    Returns:
        A LangGraphAdapter instance.

    Example:
        ```python
        from langgraph.prebuilt import create_react_agent
        from langchain_openai import ChatOpenAI
        from reminix_langgraph import wrap
        from reminix_runtime import serve

        llm = ChatOpenAI(model="gpt-4")
        graph = create_react_agent(llm, tools=[])
        agent = wrap(graph, name="my-agent")
        serve([agent], port=8080)
        ```
    """
    return LangGraphAdapter(graph, name=name)
