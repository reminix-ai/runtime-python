"""LangGraph adapter for Reminix Runtime."""

from typing import Any

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)

from reminix_runtime import (
    BaseAdapter,
    InvokeRequest,
    InvokeResponse,
    ChatRequest,
    ChatResponse,
    Message,
)


class LangGraphAdapter(BaseAdapter):
    """Adapter for LangGraph compiled graphs."""

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
        content = message.content

        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        elif role == "tool":
            return ToolMessage(content=content, tool_call_id="unknown")
        else:
            return HumanMessage(content=content)

    def _to_reminix_message(self, message: BaseMessage) -> dict[str, str]:
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

        Args:
            request: The invoke request with messages.

        Returns:
            The invoke response with the agent's reply.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in request.messages]

        # Call the graph with state dict format
        result = await self._graph.ainvoke({"messages": lc_messages})

        # Extract messages from result
        result_messages: list[BaseMessage] = result.get("messages", [])

        # Get content from the last AI message
        content = self._get_last_ai_content(result_messages)

        # Convert all messages back to Reminix format
        response_messages = [self._to_reminix_message(m) for m in result_messages]

        return InvokeResponse(content=content, messages=response_messages)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with the agent's reply.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in request.messages]

        # Call the graph with state dict format
        result = await self._graph.ainvoke({"messages": lc_messages})

        # Extract messages from result
        result_messages: list[BaseMessage] = result.get("messages", [])

        # Get content from the last AI message
        content = self._get_last_ai_content(result_messages)

        # Convert all messages back to Reminix format
        response_messages = [self._to_reminix_message(m) for m in result_messages]

        return ChatResponse(content=content, messages=response_messages)


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
