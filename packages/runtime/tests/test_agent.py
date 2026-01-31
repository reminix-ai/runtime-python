"""Tests for the decorator-based Agent class and agent decorators."""

import pytest

from reminix_runtime import (
    Agent,
    AgentInvokeRequest,
    AgentInvokeResponse,
    Message,
    agent,
    chat_agent,
)


class TestAgentCreation:
    """Tests for Agent instantiation."""

    def test_agent_can_be_instantiated(self):
        """Agent is concrete and can be instantiated directly."""
        test_agent = Agent("my-agent")
        assert test_agent.name == "my-agent"

    def test_agent_with_metadata(self):
        """Agent can be created with custom metadata."""
        test_agent = Agent("my-agent", metadata={"version": "1.0", "author": "test"})
        assert test_agent.metadata["version"] == "1.0"
        assert test_agent.metadata["author"] == "test"

    def test_agent_default_metadata(self):
        """Agent has default metadata with capabilities, input, and output."""
        test_agent = Agent("my-agent")
        assert test_agent.metadata["capabilities"]["streaming"] is False
        assert test_agent.metadata["input"]["properties"]["prompt"]["type"] == "string"
        assert test_agent.metadata["output"]["type"] == "string"


class TestAgentHandlerRegistration:
    """Tests for handler registration with decorators."""

    def test_handler_registers_handler(self):
        """handler decorator registers the handler."""
        test_agent = Agent("test-agent")

        @test_agent.handler
        async def handle_invoke(request: AgentInvokeRequest) -> AgentInvokeResponse:
            return AgentInvokeResponse(output="test")

        # Handler should be registered
        assert test_agent._invoke_handler is not None

    def test_stream_handler_registers_handler(self):
        """stream_handler decorator registers the handler."""
        test_agent = Agent("test-agent")

        @test_agent.stream_handler
        async def handle_stream(request: AgentInvokeRequest):
            yield "test"

        # Handler should be registered
        assert test_agent._invoke_stream_handler is not None

    def test_decorator_returns_original_function(self):
        """Decorator returns the original function for reuse."""
        test_agent = Agent("test-agent")

        @test_agent.handler
        async def handle_invoke(request: AgentInvokeRequest) -> AgentInvokeResponse:
            return AgentInvokeResponse(output="test")

        # The decorated function should be returned
        assert handle_invoke is not None
        assert callable(handle_invoke)


class TestAgentStreamingFlags:
    """Tests for streaming capability detection."""

    def test_streaming_false_by_default(self):
        """streaming is False when no stream handler is registered."""
        test_agent = Agent("test-agent")
        assert test_agent.metadata["capabilities"]["streaming"] is False

    def test_streaming_true_when_handler_registered(self):
        """streaming is True when stream handler is registered."""
        test_agent = Agent("test-agent")

        @test_agent.stream_handler
        async def handle_stream(request: AgentInvokeRequest):
            yield "test"

        assert test_agent.metadata["capabilities"]["streaming"] is True


class TestAgentInvoke:
    """Tests for invoke functionality."""

    @pytest.mark.asyncio
    async def test_invoke_calls_registered_handler(self):
        """invoke calls the registered handler."""
        test_agent = Agent("test-agent")

        @test_agent.handler
        async def handle_invoke(request: AgentInvokeRequest) -> AgentInvokeResponse:
            task = request.input.get("task", "unknown")
            return {"output": f"Completed: {task}"}

        request = AgentInvokeRequest(input={"task": "summarize"})
        response = await test_agent.invoke(request)

        assert response["output"] == "Completed: summarize"

    @pytest.mark.asyncio
    async def test_invoke_without_handler_raises(self):
        """invoke raises NotImplementedError when no handler is registered."""
        test_agent = Agent("test-agent")
        request = AgentInvokeRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError) as exc_info:
            await test_agent.invoke(request)

        assert "No invoke handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentInvokeStream:
    """Tests for streaming invoke functionality."""

    @pytest.mark.asyncio
    async def test_invoke_stream_calls_registered_handler(self):
        """invoke_stream calls the registered handler."""
        test_agent = Agent("test-agent")

        @test_agent.stream_handler
        async def handle_stream(request: AgentInvokeRequest):
            yield "Hello"
            yield " world"

        request = AgentInvokeRequest(input={"task": "test"})
        chunks = []
        async for chunk in test_agent.invoke_stream(request):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_invoke_stream_without_handler_raises(self):
        """invoke_stream raises NotImplementedError when no handler is registered."""
        test_agent = Agent("test-agent")
        request = AgentInvokeRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError) as exc_info:
            async for _ in test_agent.invoke_stream(request):
                pass

        assert "No streaming invoke handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentWithContext:
    """Tests for context handling."""

    @pytest.mark.asyncio
    async def test_invoke_handler_receives_context(self):
        """invoke handler receives context from request."""
        test_agent = Agent("test-agent")
        received_context = None

        @test_agent.handler
        async def handle_invoke(request: AgentInvokeRequest) -> AgentInvokeResponse:
            nonlocal received_context
            received_context = request.context
            return AgentInvokeResponse(output="done")

        request = AgentInvokeRequest(
            input={"task": "test"}, context={"user_id": "123", "session": "abc"}
        )
        await test_agent.invoke(request)

        assert received_context == {"user_id": "123", "session": "abc"}


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

    def test_agent_decorator_extracts_input(self):
        """@agent decorator extracts input schema from function signature."""

        @agent
        async def my_agent(name: str, count: int = 5) -> str:
            """Test agent."""
            return name * count

        input_schema = my_agent.metadata["input"]
        assert input_schema["type"] == "object"
        assert "name" in input_schema["properties"]
        assert "count" in input_schema["properties"]
        assert "name" in input_schema["required"]
        assert "count" not in input_schema["required"]
        assert input_schema["properties"]["count"]["default"] == 5

    def test_agent_decorator_extracts_output(self):
        """@agent decorator extracts output schema from return type."""

        @agent
        async def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        output = calculator.metadata.get("output")
        assert output is not None
        # Default output is string for simple return types wrapped

    def test_agent_decorator_default_output(self):
        """@agent decorator has default output schema."""

        @agent
        async def processor(x: str):
            """Process something."""
            return x

        assert processor.metadata["output"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_agent_decorator_invoke(self):
        """@agent decorated function can handle invoke requests."""

        @agent
        async def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        request = AgentInvokeRequest(input={"a": 3, "b": 4})
        response = await calculator.invoke(request)

        assert response["output"] == 7.0

    @pytest.mark.asyncio
    async def test_agent_decorator_sync_function(self):
        """@agent decorator works with sync functions."""

        @agent
        def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        request = AgentInvokeRequest(input={"a": 5, "b": 3})
        response = await calculator.invoke(request)

        assert response["output"] == 8.0

    @pytest.mark.asyncio
    async def test_agent_decorator_streaming(self):
        """@agent decorator works with async generators for streaming."""

        @agent
        async def streamer(text: str):
            """Stream text word by word."""
            for word in text.split():
                yield word + " "

        # Test streaming
        assert streamer.metadata["capabilities"]["streaming"] is True

        request = AgentInvokeRequest(input={"text": "hello world"})
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

        request = AgentInvokeRequest(input={"text": "hello world"})
        response = await streamer.invoke(request)

        assert response["output"] == "hello world "


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

    def test_chat_agent_decorator_has_input_schema(self):
        """@chat_agent decorator sets standard input schema."""

        @chat_agent
        async def assistant(messages: list[Message]) -> str:
            """A helpful assistant."""
            return "Hello!"

        input_schema = assistant.metadata["input"]
        assert input_schema["type"] == "object"
        assert "messages" in input_schema["properties"]
        assert input_schema["properties"]["messages"]["type"] == "array"
        assert "messages" in input_schema["required"]

    def test_chat_agent_decorator_has_output_schema(self):
        """@chat_agent decorator sets standard output schema."""

        @chat_agent
        async def assistant(messages: list[Message]) -> list[Message]:
            """A helpful assistant."""
            return [Message(role="assistant", content="Hello!")]

        output = assistant.metadata["output"]
        assert output["type"] == "object"
        assert "messages" in output["properties"]
        assert output["properties"]["messages"]["type"] == "array"

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_invoke(self):
        """@chat_agent decorated function can handle invoke requests."""

        @chat_agent
        async def echo_bot(messages: list[Message]) -> list[Message]:
            """Echo the last message."""
            last_msg = messages[-1].content if messages else ""
            return [Message(role="assistant", content=f"You said: {last_msg}")]

        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "hello"}]})
        response = await echo_bot.invoke(request)

        assert response["output"]["messages"][0]["role"] == "assistant"
        assert response["output"]["messages"][0]["content"] == "You said: hello"

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_with_context(self):
        """@chat_agent decorated function can receive context."""

        @chat_agent
        async def contextual_bot(
            messages: list[Message], context: dict | None = None
        ) -> list[Message]:
            """Bot with context."""
            user_id = context.get("user_id") if context else "unknown"
            return [Message(role="assistant", content=f"Hello user {user_id}!")]

        request = AgentInvokeRequest(
            input={"messages": [{"role": "user", "content": "hi"}]},
            context={"user_id": "123"},
        )
        response = await contextual_bot.invoke(request)

        assert response["output"]["messages"][0]["content"] == "Hello user 123!"

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_sync_function(self):
        """@chat_agent decorator works with sync functions."""

        @chat_agent
        def simple_bot(messages: list[Message]) -> list[Message]:
            """Simple sync bot."""
            return [Message(role="assistant", content="Hello from sync!")]

        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        response = await simple_bot.invoke(request)

        assert response["output"]["messages"][0]["content"] == "Hello from sync!"

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
        assert streaming_bot.metadata["capabilities"]["streaming"] is True

        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        chunks = []
        async for chunk in streaming_bot.invoke_stream(request):
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

        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        response = await streaming_bot.invoke(request)

        assert response["output"]["messages"][0]["role"] == "assistant"
        assert response["output"]["messages"][0]["content"] == "Hello world!"
