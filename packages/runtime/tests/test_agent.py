"""Tests for the decorator-based Agent class and agent decorators."""

import pytest

from reminix_runtime import (
    Agent,
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
    Message,
    agent,
    chat_agent,
)


class TestAgentCreation:
    """Tests for Agent instantiation."""

    def test_agent_can_be_instantiated(self):
        """Agent is concrete and can be instantiated directly."""
        agent = Agent("my-agent")
        assert agent.name == "my-agent"

    def test_agent_with_metadata(self):
        """Agent can be created with custom metadata."""
        agent = Agent("my-agent", metadata={"version": "1.0", "author": "test"})
        assert agent.metadata["type"] == "agent"
        assert agent.metadata["version"] == "1.0"
        assert agent.metadata["author"] == "test"

    def test_agent_default_metadata(self):
        """Agent has default metadata with type."""
        agent = Agent("my-agent")
        assert agent.metadata == {"type": "agent"}


class TestAgentHandlerRegistration:
    """Tests for handler registration with decorators."""

    def test_on_invoke_registers_handler(self):
        """on_invoke decorator registers the handler."""
        agent = Agent("test-agent")

        @agent.on_invoke
        async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
            return InvokeResponse(output="test")

        # Handler should be registered
        assert agent._invoke_handler is not None

    def test_on_chat_registers_handler(self):
        """on_chat decorator registers the handler."""
        agent = Agent("test-agent")

        @agent.on_chat
        async def handle_chat(request: ChatRequest) -> ChatResponse:
            return ChatResponse(output="test", messages=[])

        # Handler should be registered
        assert agent._chat_handler is not None

    def test_on_invoke_stream_registers_handler(self):
        """on_invoke_stream decorator registers the handler."""
        agent = Agent("test-agent")

        @agent.on_invoke_stream
        async def handle_stream(request: InvokeRequest):
            yield '{"chunk": "test"}'

        # Handler should be registered
        assert agent._invoke_stream_handler is not None

    def test_on_chat_stream_registers_handler(self):
        """on_chat_stream decorator registers the handler."""
        agent = Agent("test-agent")

        @agent.on_chat_stream
        async def handle_stream(request: ChatRequest):
            yield '{"chunk": "test"}'

        # Handler should be registered
        assert agent._chat_stream_handler is not None

    def test_decorator_returns_original_function(self):
        """Decorator returns the original function for reuse."""
        agent = Agent("test-agent")

        @agent.on_invoke
        async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
            return InvokeResponse(output="test")

        # The decorated function should be returned
        assert handle_invoke is not None
        assert callable(handle_invoke)


class TestAgentStreamingFlags:
    """Tests for streaming capability detection."""

    def test_invoke_streaming_false_by_default(self):
        """invoke_streaming is False when no stream handler is registered."""
        agent = Agent("test-agent")
        assert agent.invoke_streaming is False

    def test_chat_streaming_false_by_default(self):
        """chat_streaming is False when no stream handler is registered."""
        agent = Agent("test-agent")
        assert agent.chat_streaming is False

    def test_invoke_streaming_true_when_handler_registered(self):
        """invoke_streaming is True when stream handler is registered."""
        agent = Agent("test-agent")

        @agent.on_invoke_stream
        async def handle_stream(request: InvokeRequest):
            yield '{"chunk": "test"}'

        assert agent.invoke_streaming is True

    def test_chat_streaming_true_when_handler_registered(self):
        """chat_streaming is True when stream handler is registered."""
        agent = Agent("test-agent")

        @agent.on_chat_stream
        async def handle_stream(request: ChatRequest):
            yield '{"chunk": "test"}'

        assert agent.chat_streaming is True


class TestAgentInvoke:
    """Tests for invoke functionality."""

    @pytest.mark.asyncio
    async def test_invoke_calls_registered_handler(self):
        """invoke calls the registered handler."""
        agent = Agent("test-agent")

        @agent.on_invoke
        async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
            task = request.input.get("task", "unknown")
            return InvokeResponse(output=f"Completed: {task}")

        request = InvokeRequest(input={"task": "summarize"})
        response = await agent.invoke(request)

        assert response.output == "Completed: summarize"

    @pytest.mark.asyncio
    async def test_invoke_without_handler_raises(self):
        """invoke raises NotImplementedError when no handler is registered."""
        agent = Agent("test-agent")
        request = InvokeRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError) as exc_info:
            await agent.invoke(request)

        assert "No invoke handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentChat:
    """Tests for chat functionality."""

    @pytest.mark.asyncio
    async def test_chat_calls_registered_handler(self):
        """chat calls the registered handler."""
        agent = Agent("test-agent")

        @agent.on_chat
        async def handle_chat(request: ChatRequest) -> ChatResponse:
            user_msg = request.messages[-1].content
            return ChatResponse(
                output=f"Hello: {user_msg}",
                messages=[{"role": "assistant", "content": f"Hello: {user_msg}"}],
            )

        request = ChatRequest(messages=[{"role": "user", "content": "hi"}])
        response = await agent.chat(request)

        assert response.output == "Hello: hi"
        assert len(response.messages) == 1

    @pytest.mark.asyncio
    async def test_chat_without_handler_raises(self):
        """chat raises NotImplementedError when no handler is registered."""
        agent = Agent("test-agent")
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}])

        with pytest.raises(NotImplementedError) as exc_info:
            await agent.chat(request)

        assert "No chat handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentInvokeStream:
    """Tests for streaming invoke functionality."""

    @pytest.mark.asyncio
    async def test_invoke_stream_calls_registered_handler(self):
        """invoke_stream calls the registered handler."""
        agent = Agent("test-agent")

        @agent.on_invoke_stream
        async def handle_stream(request: InvokeRequest):
            yield '{"chunk": "Hello"}'
            yield '{"chunk": " world"}'

        request = InvokeRequest(input={"task": "test"})
        chunks = []
        async for chunk in agent.invoke_stream(request):
            chunks.append(chunk)

        assert chunks == ['{"chunk": "Hello"}', '{"chunk": " world"}']

    @pytest.mark.asyncio
    async def test_invoke_stream_without_handler_raises(self):
        """invoke_stream raises NotImplementedError when no handler is registered."""
        agent = Agent("test-agent")
        request = InvokeRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError) as exc_info:
            async for _ in agent.invoke_stream(request):
                pass

        assert "No streaming invoke handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentChatStream:
    """Tests for streaming chat functionality."""

    @pytest.mark.asyncio
    async def test_chat_stream_calls_registered_handler(self):
        """chat_stream calls the registered handler."""
        agent = Agent("test-agent")

        @agent.on_chat_stream
        async def handle_stream(request: ChatRequest):
            yield '{"chunk": "Hi"}'
            yield '{"chunk": " there"}'

        request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
        chunks = []
        async for chunk in agent.chat_stream(request):
            chunks.append(chunk)

        assert chunks == ['{"chunk": "Hi"}', '{"chunk": " there"}']

    @pytest.mark.asyncio
    async def test_chat_stream_without_handler_raises(self):
        """chat_stream raises NotImplementedError when no handler is registered."""
        agent = Agent("test-agent")
        request = ChatRequest(messages=[{"role": "user", "content": "hi"}])

        with pytest.raises(NotImplementedError) as exc_info:
            async for _ in agent.chat_stream(request):
                pass

        assert "No streaming chat handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentWithContext:
    """Tests for context handling."""

    @pytest.mark.asyncio
    async def test_invoke_handler_receives_context(self):
        """invoke handler receives context from request."""
        agent = Agent("test-agent")
        received_context = None

        @agent.on_invoke
        async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
            nonlocal received_context
            received_context = request.context
            return InvokeResponse(output="done")

        request = InvokeRequest(
            input={"task": "test"}, context={"user_id": "123", "session": "abc"}
        )
        await agent.invoke(request)

        assert received_context == {"user_id": "123", "session": "abc"}

    @pytest.mark.asyncio
    async def test_chat_handler_receives_context(self):
        """chat handler receives context from request."""
        agent = Agent("test-agent")
        received_context = None

        @agent.on_chat
        async def handle_chat(request: ChatRequest) -> ChatResponse:
            nonlocal received_context
            received_context = request.context
            return ChatResponse(output="done", messages=[])

        request = ChatRequest(
            messages=[{"role": "user", "content": "hi"}], context={"user_id": "456"}
        )
        await agent.chat(request)

        assert received_context == {"user_id": "456"}


# =============================================================================
# Tests for @agent decorator
# =============================================================================


class TestAgentDecorator:
    """Tests for the @agent decorator."""

    def test_agent_decorator_creates_agent(self):
        """@agent decorator creates an Agent instance."""

        @agent
        async def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        assert isinstance(calculator, Agent)
        assert calculator.name == "calculator"

    def test_agent_decorator_with_custom_name(self):
        """@agent decorator accepts custom name."""

        @agent(name="custom-calc")
        async def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        assert calculator.name == "custom-calc"

    def test_agent_decorator_extracts_description(self):
        """@agent decorator extracts description from docstring."""

        @agent
        async def my_agent(x: str) -> str:
            """This is the agent description."""
            return x

        assert my_agent.metadata["description"] == "This is the agent description."

    def test_agent_decorator_with_custom_description(self):
        """@agent decorator accepts custom description."""

        @agent(description="Custom description")
        async def my_agent(x: str) -> str:
            """Original docstring."""
            return x

        assert my_agent.metadata["description"] == "Custom description"

    def test_agent_decorator_extracts_parameters(self):
        """@agent decorator extracts parameters from function signature."""

        @agent
        async def my_agent(name: str, count: int = 5) -> str:
            """Test agent."""
            return name * count

        params = my_agent.metadata["parameters"]
        assert params["type"] == "object"
        assert "name" in params["properties"]
        assert "count" in params["properties"]
        assert "name" in params["required"]
        assert "count" not in params["required"]
        assert params["properties"]["count"]["default"] == 5

    @pytest.mark.asyncio
    async def test_agent_decorator_invoke(self):
        """@agent decorated function can handle invoke requests."""

        @agent
        async def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        request = InvokeRequest(input={"a": 3, "b": 4})
        response = await calculator.invoke(request)

        assert response.output == 7.0

    @pytest.mark.asyncio
    async def test_agent_decorator_sync_function(self):
        """@agent decorator works with sync functions."""

        @agent
        def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        request = InvokeRequest(input={"a": 5, "b": 3})
        response = await calculator.invoke(request)

        assert response.output == 8.0

    @pytest.mark.asyncio
    async def test_agent_decorator_streaming(self):
        """@agent decorator works with async generators for streaming."""

        @agent
        async def streamer(text: str):
            """Stream text word by word."""
            for word in text.split():
                yield word + " "

        # Test streaming
        assert streamer.invoke_streaming is True

        request = InvokeRequest(input={"text": "hello world"})
        chunks = []
        async for chunk in streamer.invoke_stream(request):
            chunks.append(chunk)

        assert chunks == ["hello ", "world "]

    @pytest.mark.asyncio
    async def test_agent_decorator_streaming_non_stream_request(self):
        """@agent streaming decorator collects chunks for non-streaming requests."""

        @agent
        async def streamer(text: str):
            """Stream text word by word."""
            for word in text.split():
                yield word + " "

        request = InvokeRequest(input={"text": "hello world"})
        response = await streamer.invoke(request)

        assert response.output == "hello world "


# =============================================================================
# Tests for @chat_agent decorator
# =============================================================================


class TestChatAgentDecorator:
    """Tests for the @chat_agent decorator."""

    def test_chat_agent_decorator_creates_agent(self):
        """@chat_agent decorator creates an Agent instance."""

        @chat_agent
        async def assistant(messages: list[Message]) -> str:
            """A helpful assistant."""
            return "Hello!"

        assert isinstance(assistant, Agent)
        assert assistant.name == "assistant"

    def test_chat_agent_decorator_with_custom_name(self):
        """@chat_agent decorator accepts custom name."""

        @chat_agent(name="my-bot")
        async def assistant(messages: list[Message]) -> str:
            """A helpful assistant."""
            return "Hello!"

        assert assistant.name == "my-bot"

    def test_chat_agent_decorator_extracts_description(self):
        """@chat_agent decorator extracts description from docstring."""

        @chat_agent
        async def assistant(messages: list[Message]) -> str:
            """This is a helpful assistant."""
            return "Hello!"

        assert assistant.metadata["description"] == "This is a helpful assistant."

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_chat(self):
        """@chat_agent decorated function can handle chat requests."""

        @chat_agent
        async def echo_bot(messages: list[Message]) -> str:
            """Echo the last message."""
            last_msg = messages[-1].content if messages else ""
            return f"You said: {last_msg}"

        request = ChatRequest(messages=[Message(role="user", content="hello")])
        response = await echo_bot.chat(request)

        assert response.output == "You said: hello"
        assert len(response.messages) == 2
        assert response.messages[-1]["role"] == "assistant"
        assert response.messages[-1]["content"] == "You said: hello"

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_with_context(self):
        """@chat_agent decorated function can receive context."""

        @chat_agent
        async def contextual_bot(messages: list[Message], context: dict | None = None) -> str:
            """Bot with context."""
            user_id = context.get("user_id") if context else "unknown"
            return f"Hello user {user_id}!"

        request = ChatRequest(
            messages=[Message(role="user", content="hi")],
            context={"user_id": "123"},
        )
        response = await contextual_bot.chat(request)

        assert response.output == "Hello user 123!"

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_sync_function(self):
        """@chat_agent decorator works with sync functions."""

        @chat_agent
        def simple_bot(messages: list[Message]) -> str:
            """Simple sync bot."""
            return "Hello from sync!"

        request = ChatRequest(messages=[Message(role="user", content="hi")])
        response = await simple_bot.chat(request)

        assert response.output == "Hello from sync!"

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_streaming(self):
        """@chat_agent decorator works with async generators for streaming."""

        @chat_agent
        async def streaming_bot(messages: list[Message]):
            """Streaming bot."""
            yield "Hello"
            yield " "
            yield "world"
            yield "!"

        # Test streaming flag
        assert streaming_bot.chat_streaming is True

        request = ChatRequest(messages=[Message(role="user", content="hi")])
        chunks = []
        async for chunk in streaming_bot.chat_stream(request):
            chunks.append(chunk)

        assert chunks == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_streaming_non_stream_request(self):
        """@chat_agent streaming decorator collects chunks for non-streaming requests."""

        @chat_agent
        async def streaming_bot(messages: list[Message]):
            """Streaming bot."""
            yield "Hello"
            yield " "
            yield "world"
            yield "!"

        request = ChatRequest(messages=[Message(role="user", content="hi")])
        response = await streaming_bot.chat(request)

        assert response.output == "Hello world!"
        assert response.messages[-1]["content"] == "Hello world!"
