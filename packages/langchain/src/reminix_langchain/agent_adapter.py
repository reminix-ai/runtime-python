"""LangChain agent adapter for Reminix Runtime."""

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
    AgentAdapter,
    AgentInvokeRequest,
    AgentInvokeResponseDict,
    Message,
    serve,
)


class LangChainAgentAdapter(AgentAdapter):
    """Agent adapter for LangChain agents and runnables."""

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
            tool_call_id = getattr(message, "tool_call_id", None) or "unknown"
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

    def _build_langchain_input(self, request: AgentInvokeRequest) -> Any:
        """Build LangChain input from invoke request."""
        # Check if input contains messages (chat-style)
        if "messages" in request.input:
            # Convert message dicts to LangChain messages
            lc_messages = []
            for m in request.input["messages"]:
                msg = Message(role=m.get("role", "user"), content=m.get("content", ""))
                lc_messages.append(self._to_langchain_message(msg))
            return lc_messages
        elif "prompt" in request.input:
            return request.input["prompt"]
        else:
            # Pass input directly to the runnable
            return request.input

    async def invoke(self, request: AgentInvokeRequest) -> AgentInvokeResponseDict:
        """Handle an invoke request.

        For both task-oriented and chat-style operations. Expects input with 'messages' key
        or a 'prompt' key for simple text generation.

        Args:
            request: The invoke request with input data.

        Returns:
            The invoke response with the output.
        """
        invoke_input = self._build_langchain_input(request)

        response = await self._agent.ainvoke(invoke_input)

        # Extract output from response
        if isinstance(response, BaseMessage):
            output = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
        elif isinstance(response, dict):
            output = response
        else:
            output = str(response)

        return {"output": output}

    async def invoke_stream(self, request: AgentInvokeRequest) -> AsyncIterator[str]:
        """Handle a streaming invoke request.

        Streams chunks from the LangChain runnable.

        Args:
            request: The invoke request with input data.

        Yields:
            JSON-encoded chunks from the stream.
        """
        stream_input = self._build_langchain_input(request)

        async for chunk in self._agent.astream(stream_input):
            if isinstance(chunk, (BaseMessage, AIMessageChunk)):
                content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            elif isinstance(chunk, dict):
                content = json.dumps(chunk)
            else:
                content = str(chunk)
            yield json.dumps({"chunk": content})


def wrap_agent(agent: Runnable, name: str = "langchain-agent") -> LangChainAgentAdapter:
    """Wrap a LangChain agent for use with Reminix Runtime.

    Args:
        agent: A LangChain runnable (e.g., ChatModel, chain, agent).
        name: Name for the agent.

    Returns:
        A LangChainAgentAdapter instance.

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from reminix_langchain import wrap_agent
        from reminix_runtime import serve

        llm = ChatOpenAI(model="gpt-4")
        agent = wrap_agent(llm, name="my-agent")
        serve(agents=[agent], port=8080)
        ```
    """
    return LangChainAgentAdapter(agent, name=name)


def serve_agent(
    agent: Runnable,
    name: str = "langchain-agent",
    port: int = 8080,
    host: str = "0.0.0.0",
) -> None:
    """Wrap a LangChain runnable and serve it immediately.

    This is a convenience function that combines `wrap` and `serve` for single-agent setups.

    Args:
        agent: A LangChain runnable (e.g., ChatModel, chain, agent).
        name: Name for the agent.
        port: Port to serve on.
        host: Host to bind to.

    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from reminix_langchain import serve_agent

        llm = ChatOpenAI(model="gpt-4")
        serve_agent(llm, name="my-agent", port=8080)
        ```
    """
    wrapped_agent = wrap_agent(agent, name=name)
    serve(agents=[wrapped_agent], port=port, host=host)
