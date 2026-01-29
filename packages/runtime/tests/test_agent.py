"""Tests for the decorator-based Agent class and agent decorators."""

import pytest

from reminix_runtime import (
    Agent,
    ExecuteRequest,
    ExecuteResponse,
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
        """Agent has default metadata with type, parameters, and keys."""
        agent = Agent("my-agent")
        assert agent.metadata["type"] == "agent"
        assert agent.metadata["requestKeys"] == ["prompt"]
        assert agent.metadata["responseKeys"] == ["content"]
        assert agent.metadata["parameters"]["properties"]["prompt"]["type"] == "string"


class TestAgentHandlerRegistration:
    """Tests for handler registration with decorators."""

    def test_on_execute_registers_handler(self):
        """on_execute decorator registers the handler."""
        agent = Agent("test-agent")

        @agent.handler
        async def handle_execute(request: ExecuteRequest) -> ExecuteResponse:
            return ExecuteResponse(content="test")

        # Handler should be registered
        assert agent._execute_handler is not None

    def test_stream_handler_registers_handler(self):
        """stream_handler decorator registers the handler."""
        agent = Agent("test-agent")

        @agent.stream_handler
        async def handle_stream(request: ExecuteRequest):
            yield '{"chunk": "test"}'

        # Handler should be registered
        assert agent._execute_stream_handler is not None

    def test_decorator_returns_original_function(self):
        """Decorator returns the original function for reuse."""
        agent = Agent("test-agent")

        @agent.handler
        async def handle_execute(request: ExecuteRequest) -> ExecuteResponse:
            return ExecuteResponse(content="test")

        # The decorated function should be returned
        assert handle_execute is not None
        assert callable(handle_execute)


class TestAgentStreamingFlags:
    """Tests for streaming capability detection."""

    def test_streaming_false_by_default(self):
        """streaming is False when no stream handler is registered."""
        agent = Agent("test-agent")
        assert agent.streaming is False

    def test_streaming_true_when_handler_registered(self):
        """streaming is True when stream handler is registered."""
        agent = Agent("test-agent")

        @agent.stream_handler
        async def handle_stream(request: ExecuteRequest):
            yield '{"chunk": "test"}'

        assert agent.streaming is True


class TestAgentExecute:
    """Tests for execute functionality."""

    @pytest.mark.asyncio
    async def test_execute_calls_registered_handler(self):
        """execute calls the registered handler."""
        test_agent = Agent("test-agent")

        @test_agent.on_execute
        async def handle_execute(request: ExecuteRequest) -> ExecuteResponse:
            task = request.input.get("task", "unknown")
            return {"content": f"Completed: {task}"}

        request = ExecuteRequest(input={"task": "summarize"})
        response = await test_agent.execute(request)

        assert response["content"] == "Completed: summarize"

    @pytest.mark.asyncio
    async def test_execute_without_handler_raises(self):
        """execute raises NotImplementedError when no handler is registered."""
        agent = Agent("test-agent")
        request = ExecuteRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError) as exc_info:
            await agent.execute(request)

        assert "No execute handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentExecuteStream:
    """Tests for streaming execute functionality."""

    @pytest.mark.asyncio
    async def test_execute_stream_calls_registered_handler(self):
        """execute_stream calls the registered handler."""
        agent = Agent("test-agent")

        @agent.stream_handler
        async def handle_stream(request: ExecuteRequest):
            yield '{"chunk": "Hello"}'
            yield '{"chunk": " world"}'

        request = ExecuteRequest(input={"task": "test"})
        chunks = []
        async for chunk in agent.execute_stream(request):
            chunks.append(chunk)

        assert chunks == ['{"chunk": "Hello"}', '{"chunk": " world"}']

    @pytest.mark.asyncio
    async def test_execute_stream_without_handler_raises(self):
        """execute_stream raises NotImplementedError when no handler is registered."""
        agent = Agent("test-agent")
        request = ExecuteRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError) as exc_info:
            async for _ in agent.execute_stream(request):
                pass

        assert "No streaming execute handler registered" in str(exc_info.value)
        assert "test-agent" in str(exc_info.value)


class TestAgentWithContext:
    """Tests for context handling."""

    @pytest.mark.asyncio
    async def test_execute_handler_receives_context(self):
        """execute handler receives context from request."""
        agent = Agent("test-agent")
        received_context = None

        @agent.handler
        async def handle_execute(request: ExecuteRequest) -> ExecuteResponse:
            nonlocal received_context
            received_context = request.context
            return ExecuteResponse(content="done")

        request = ExecuteRequest(
            input={"task": "test"}, context={"user_id": "123", "session": "abc"}
        )
        await agent.execute(request)

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

    def test_agent_decorator_extracts_output(self):
        """@agent decorator extracts output schema from return type and wraps it."""

        @agent
        async def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        output = calculator.metadata.get("output")
        assert output is not None
        assert output["type"] == "object"
        assert "content" in output["properties"]
        assert output["properties"]["content"]["type"] == "number"
        assert output["required"] == ["content"]

    def test_agent_decorator_output_dict_type(self):
        """@agent decorator handles dict return type and wraps it."""

        @agent
        async def get_data(key: str) -> dict:
            """Get data."""
            return {"key": key}

        output = get_data.metadata.get("output")
        assert output is not None
        assert output["type"] == "object"
        assert "content" in output["properties"]
        assert output["properties"]["content"]["type"] == "object"
        assert output["required"] == ["content"]

    def test_agent_decorator_no_output_when_no_return_type(self):
        """@agent decorator omits output when no return type hint."""

        @agent
        async def processor(x: str):
            """Process something."""
            return x

        assert "output" not in processor.metadata

    @pytest.mark.asyncio
    async def test_agent_decorator_execute(self):
        """@agent decorated function can handle execute requests."""

        @agent
        async def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        request = ExecuteRequest(input={"a": 3, "b": 4})
        response = await calculator.execute(request)

        assert response["content"] == 7.0

    @pytest.mark.asyncio
    async def test_agent_decorator_sync_function(self):
        """@agent decorator works with sync functions."""

        @agent
        def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        request = ExecuteRequest(input={"a": 5, "b": 3})
        response = await calculator.execute(request)

        assert response["content"] == 8.0

    @pytest.mark.asyncio
    async def test_agent_decorator_streaming(self):
        """@agent decorator works with async generators for streaming."""

        @agent
        async def streamer(text: str):
            """Stream text word by word."""
            for word in text.split():
                yield word + " "

        # Test streaming
        assert streamer.streaming is True

        request = ExecuteRequest(input={"text": "hello world"})
        chunks = []
        async for chunk in streamer.execute_stream(request):
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

        request = ExecuteRequest(input={"text": "hello world"})
        response = await streamer.execute(request)

        assert response["content"] == "hello world "


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

    def test_chat_agent_decorator_has_parameters_schema(self):
        """@chat_agent decorator sets standard parameters schema."""

        @chat_agent
        async def assistant(messages: list[Message]) -> str:
            """A helpful assistant."""
            return "Hello!"

        params = assistant.metadata["parameters"]
        assert params["type"] == "object"
        assert "messages" in params["properties"]
        assert params["properties"]["messages"]["type"] == "array"
        assert "messages" in params["required"]

    def test_chat_agent_decorator_has_output_schema(self):
        """@chat_agent decorator sets standard output schema wrapped for response structure."""

        @chat_agent
        async def assistant(messages: list[Message]) -> list[Message]:
            """A helpful assistant."""
            return [Message(role="assistant", content="Hello!")]

        output = assistant.metadata["output"]
        assert output["type"] == "object"
        assert "messages" in output["properties"]
        assert output["properties"]["messages"]["type"] == "array"
        assert "role" in output["properties"]["messages"]["items"]["properties"]
        assert "content" in output["properties"]["messages"]["items"]["properties"]
        assert output["required"] == ["messages"]

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_execute(self):
        """@chat_agent decorated function can handle execute requests."""

        @chat_agent
        async def echo_bot(messages: list[Message]) -> list[Message]:
            """Echo the last message."""
            last_msg = messages[-1].content if messages else ""
            return [Message(role="assistant", content=f"You said: {last_msg}")]

        request = ExecuteRequest(input={"messages": [{"role": "user", "content": "hello"}]})
        response = await echo_bot.execute(request)

        assert response["messages"][0]["role"] == "assistant"
        assert response["messages"][0]["content"] == "You said: hello"

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

        request = ExecuteRequest(
            input={"messages": [{"role": "user", "content": "hi"}]},
            context={"user_id": "123"},
        )
        response = await contextual_bot.execute(request)

        assert response["messages"][0]["content"] == "Hello user 123!"

    @pytest.mark.asyncio
    async def test_chat_agent_decorator_sync_function(self):
        """@chat_agent decorator works with sync functions."""

        @chat_agent
        def simple_bot(messages: list[Message]) -> list[Message]:
            """Simple sync bot."""
            return [Message(role="assistant", content="Hello from sync!")]

        request = ExecuteRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        response = await simple_bot.execute(request)

        assert response["messages"][0]["content"] == "Hello from sync!"

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
        assert streaming_bot.streaming is True

        request = ExecuteRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        chunks = []
        async for chunk in streaming_bot.execute_stream(request):
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

        request = ExecuteRequest(input={"messages": [{"role": "user", "content": "hi"}]})
        response = await streaming_bot.execute(request)

        assert response["messages"][0]["role"] == "assistant"
        assert response["messages"][0]["content"] == "Hello world!"
