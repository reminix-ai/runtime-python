"""LangChain adapter for Reminix Runtime."""

import json
from typing import Any, AsyncIterator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    AIMessageChunk,
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

    adapter_name = "langchain"

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
        content = message.content or ""

        if role == "user":
            return HumanMessage(content=content)
        elif role == "assistant":
            return AIMessage(content=content)
        elif role == "system":
            return SystemMessage(content=content)
        elif role == "tool":
            # Tool messages require a tool_call_id
            tool_call_id = getattr(message, 'tool_call_id', None) or "unknown"
            return ToolMessage(content=content, tool_call_id=tool_call_id)
        else:
            # Fallback to HumanMessage for unknown roles
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

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        """Handle an invoke request.

        For task-oriented operations. Passes the input directly to the runnable.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        # Pass input directly to the runnable
        response = await self._agent.ainvoke(request.input)

        # Extract output from response
        if isinstance(response, BaseMessage):
            output = response.content if isinstance(response.content, str) else str(response.content)
        elif isinstance(response, dict):
            output = response
        else:
            output = str(response)

        return InvokeResponse(output=output)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Handle a chat request.

        For conversational interactions. Converts messages to LangChain format.

        Args:
            request: The chat request with messages.

        Returns:
            The chat response with output and messages.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in request.messages]

        # Call the runnable
        response = await self._agent.ainvoke(lc_messages)

        # Extract content from response
        if isinstance(response, BaseMessage):
            content = response.content if isinstance(response.content, str) else str(response.content)
            response_message = self._to_reminix_message(response)
        else:
            content = str(response)
            response_message = {"role": "assistant", "content": content}

        # Build response messages (original + assistant response)
        response_messages: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        response_messages.append(response_message)

        return ChatResponse(output=content, messages=response_messages)

    async def invoke_stream(self, request: InvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request.

        Streams chunks from the LangChain runnable.

        Args:
            request: The invoke request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        async for chunk in self._agent.astream(request.input):
            if isinstance(chunk, BaseMessage):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            elif isinstance(chunk, dict):
                content = json.dumps(chunk)
            else:
                content = str(chunk)
            yield json.dumps({"chunk": content})

    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]:
        """Handle a streaming chat request.

        Streams chunks from the LangChain runnable.

        Args:
            request: The chat request with messages.

        Yields:
            JSON-encoded chunks from the stream.
        """
        # Convert messages to LangChain format
        lc_messages = [self._to_langchain_message(m) for m in request.messages]

        # Stream from the runnable
        async for chunk in self._agent.astream(lc_messages):
            if isinstance(chunk, (BaseMessage, AIMessageChunk)):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            elif isinstance(chunk, dict):
                content = json.dumps(chunk)
            else:
                content = str(chunk)
            yield json.dumps({"chunk": content})


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
