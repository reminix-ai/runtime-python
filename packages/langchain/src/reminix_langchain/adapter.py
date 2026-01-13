"""LangChain adapter for Reminix Runtime."""

from typing import Any, AsyncIterator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable

from reminix_runtime import (
    BaseAdapter,
    InvokeRequest,
    InvokeResponse,
    ChatRequest,
    ChatResponse,
    Message,
)


class LangChainAdapter(BaseAdapter):
    """Adapter for LangChain agents and runnables."""

    def __init__(self, agent: Runnable, name: str = "langchain-agent") -> None:
        """Initialize the adapter.

        Args:
            agent: A LangChain runnable (e.g., ChatModel, chain, agent).
            name: Name for the agent.
        """
        self._agent = agent
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
            # Tool messages require a tool_call_id, use a placeholder if not provided
            return ToolMessage(content=content, tool_call_id="unknown")
        else:
            # Fallback to HumanMessage for unknown roles
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

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request.

        Args:
            request: The invoke request with messages.

        Returns:
            The invoke response with the agent's reply.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in request.messages]

        # Call the runnable
        response = await self._agent.ainvoke(lc_messages)

        # Extract content from response
        if isinstance(response, BaseMessage):
            content = response.content if isinstance(response.content, str) else str(response.content)
        else:
            content = str(response)

        # Build response messages (original + assistant response)
        response_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append({"role": "assistant", "content": content})

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

        # Call the runnable
        response = await self._agent.ainvoke(lc_messages)

        # Extract content from response
        if isinstance(response, BaseMessage):
            content = response.content if isinstance(response.content, str) else str(response.content)
        else:
            content = str(response)

        # Build response messages (original + assistant response)
        response_messages = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append({"role": "assistant", "content": content})

        return ChatResponse(content=content, messages=response_messages)


def wrap(agent: Runnable, name: str = "langchain-agent") -> LangChainAdapter:
    """Wrap a LangChain agent for use with Reminix Runtime.

    Args:
        agent: A LangChain runnable (e.g., ChatModel, chain, agent).
        name: Name for the agent.

    Returns:
        A LangChainAdapter instance.

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from reminix_langchain import wrap
        from reminix_runtime import serve

        llm = ChatOpenAI(model="gpt-4")
        agent = wrap(llm, name="my-agent")
        serve([agent], port=8080)
        ```
    """
    return LangChainAdapter(agent, name=name)
