"""Tests for the decorator-based Agent class and agent decorators."""

import pytest

from reminix_runtime import (
    Agent,
    AgentInvokeRequest,
    AgentInvokeResponse,
    agent,
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


class TestAgentTemplates:
    """Tests for @agent(template=...) templates (prompt, chat, task)."""

    def test_template_prompt_metadata(self):
        """template=prompt sets prompt input and string output in metadata."""

        @agent(template="prompt")
        async def echo(prompt: str):
            return f"You said: {prompt}"

        assert echo.metadata["template"] == "prompt"
        assert echo.metadata["input"]["required"] == ["prompt"]
        assert echo.metadata["input"]["properties"]["prompt"]["type"] == "string"
        assert echo.metadata["output"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_template_prompt_invoke(self):
        """template=prompt agent handles invoke with prompt input."""

        @agent(template="prompt")
        async def echo(prompt: str):
            return f"You said: {prompt}"

        request = AgentInvokeRequest(input={"prompt": "hello"})
        response = await echo.invoke(request)
        assert response["output"] == "You said: hello"

    def test_template_chat_metadata(self):
        """template=chat sets messages input and string output in metadata."""

        @agent(template="chat")
        async def chat_handler(messages: list):
            return "ok"

        assert chat_handler.metadata["template"] == "chat"
        assert chat_handler.metadata["input"]["required"] == ["messages"]
        assert "messages" in chat_handler.metadata["input"]["properties"]
        assert chat_handler.metadata["output"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_template_chat_invoke(self):
        """template=chat agent handles invoke with messages input."""

        @agent(template="chat")
        async def chat_handler(messages: list):
            last = messages[-1] if messages else {}
            return f"Reply to: {last.get('content', '')}"

        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "Hi"}]})
        response = await chat_handler.invoke(request)
        assert response["output"] == "Reply to: Hi"

    def test_template_task_metadata(self):
        """template=task sets task input and structured output in metadata."""

        @agent(template="task")
        async def task_handler(task: str, text: str | None = None):
            return f"Task {task}"

        assert task_handler.metadata["template"] == "task"
        assert task_handler.metadata["input"]["required"] == ["task"]
        assert "task" in task_handler.metadata["input"]["properties"]
        assert "description" in task_handler.metadata["output"]
        assert "Structured JSON" in task_handler.metadata["output"]["description"]

    @pytest.mark.asyncio
    async def test_template_task_invoke(self):
        """template=task agent handles invoke with task input."""

        @agent(template="task")
        async def task_handler(task: str, text: str | None = None):
            return f'Task "{task}" on: {text or "—"}'

        request = AgentInvokeRequest(input={"task": "summarize", "text": "Some content"})
        response = await task_handler.invoke(request)
        assert response["output"] == 'Task "summarize" on: Some content'

    def test_no_template_derives_from_function(self):
        """Without template, input/output are derived from function signature."""

        @agent
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert "template" not in add.metadata
        assert "a" in add.metadata["input"]["properties"]
        assert "b" in add.metadata["input"]["properties"]
        assert add.metadata["input"]["required"] == ["a", "b"]

    def test_template_rag_metadata(self):
        """template=rag sets query input and string output in metadata."""

        @agent(template="rag")
        async def rag_handler(query: str):
            return f"Answer for: {query}"

        assert rag_handler.metadata["template"] == "rag"
        assert rag_handler.metadata["input"]["required"] == ["query"]
        assert "query" in rag_handler.metadata["input"]["properties"]
        assert rag_handler.metadata["output"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_template_rag_invoke(self):
        """template=rag agent handles invoke with query input."""

        @agent(template="rag")
        async def rag_handler(query: str):
            return f"Answer for: {query}"

        request = AgentInvokeRequest(input={"query": "What is X?"})
        response = await rag_handler.invoke(request)
        assert response["output"] == "Answer for: What is X?"

    def test_template_thread_metadata(self):
        """template=thread sets messages input and messages output (array) in metadata."""

        @agent(template="thread")
        async def thread_handler(messages: list):
            return messages + [{"role": "assistant", "content": "ok"}]

        assert thread_handler.metadata["template"] == "thread"
        assert thread_handler.metadata["input"]["required"] == ["messages"]
        assert "messages" in thread_handler.metadata["input"]["properties"]
        assert thread_handler.metadata["output"]["type"] == "array"
        assert "items" in thread_handler.metadata["output"]

    @pytest.mark.asyncio
    async def test_template_thread_invoke(self):
        """template=thread agent returns updated message thread (output is messages)."""

        @agent(template="thread")
        async def thread_handler(messages: list):
            last = messages[-1] if messages else {}
            return messages + [
                {"role": "assistant", "content": f"Reply: {last.get('content', '')}"}
            ]

        request = AgentInvokeRequest(input={"messages": [{"role": "user", "content": "Hello"}]})
        response = await thread_handler.invoke(request)
        output = response["output"]
        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0] == {"role": "user", "content": "Hello"}
        assert output[1]["role"] == "assistant"
        assert output[1]["content"] == "Reply: Hello"
