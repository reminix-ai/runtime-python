"""Tests for the Agent class and @agent decorator."""

import pytest

from reminix_runtime import (
    Agent,
    AgentRequest,
    agent,
)

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

        request = AgentRequest(input={"a": 3, "b": 4})
        response = await calculator.invoke(request)

        assert response["output"] == 7.0

    @pytest.mark.asyncio
    async def test_agent_decorator_sync_function(self):
        """@agent decorator works with sync functions."""

        @agent
        def calculator(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        request = AgentRequest(input={"a": 5, "b": 3})
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

        request = AgentRequest(input={"text": "hello world"})
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

        request = AgentRequest(input={"text": "hello world"})
        response = await streamer.invoke(request)

        assert response["output"] == "hello world "

    @pytest.mark.asyncio
    async def test_agent_decorator_receives_context(self):
        """@agent decorated function with context param receives request context."""

        @agent
        async def with_context(prompt: str, context: dict | None = None) -> str:
            """Echo prompt with user from context."""
            user = (context or {}).get("user_id", "anonymous")
            return f"{user}: {prompt}"

        request = AgentRequest(
            input={"prompt": "hello"},
            context={"user_id": "u-123", "tenant": "acme"},
        )
        response = await with_context.invoke(request)
        assert response["output"] == "u-123: hello"

    @pytest.mark.asyncio
    async def test_agent_decorator_context_optional(self):
        """@agent with context param works when context is not in request."""

        @agent
        async def with_context(prompt: str, context: dict | None = None) -> str:
            user = (context or {}).get("user_id", "anonymous")
            return f"{user}: {prompt}"

        request = AgentRequest(input={"prompt": "hi"})
        response = await with_context.invoke(request)
        assert response["output"] == "anonymous: hi"

    @pytest.mark.asyncio
    async def test_agent_decorator_streaming_with_context(self):
        """@agent streaming with context param receives request context."""

        @agent
        async def stream_with_context(text: str, context: dict | None = None):
            """Stream with prefix from context."""
            prefix = (context or {}).get("prefix", "")
            for word in text.split():
                yield prefix + word + " "

        request = AgentRequest(
            input={"text": "a b"},
            context={"prefix": ">"},
        )
        chunks = []
        async for chunk in stream_with_context.invoke_stream(request):
            chunks.append(chunk)
        assert chunks == [">a ", ">b "]


class TestAgentTypes:
    """Tests for @agent(type=...) types (prompt, chat, task)."""

    def test_type_prompt_metadata(self):
        """type=prompt sets prompt input and string output in metadata."""

        @agent(type="prompt")
        async def echo(prompt: str):
            return f"You said: {prompt}"

        assert echo.metadata["type"] == "prompt"
        assert echo.metadata["input"]["required"] == ["prompt"]
        assert echo.metadata["input"]["properties"]["prompt"]["type"] == "string"
        assert echo.metadata["output"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_type_prompt_invoke(self):
        """type=prompt agent handles invoke with prompt input."""

        @agent(type="prompt")
        async def echo(prompt: str):
            return f"You said: {prompt}"

        request = AgentRequest(input={"prompt": "hello"})
        response = await echo.invoke(request)
        assert response["output"] == "You said: hello"

    def test_type_chat_metadata(self):
        """type=chat sets messages input and string output in metadata."""

        @agent(type="chat")
        async def chat_handler(messages: list):
            return "ok"

        assert chat_handler.metadata["type"] == "chat"
        assert chat_handler.metadata["input"]["required"] == ["messages"]
        assert "messages" in chat_handler.metadata["input"]["properties"]
        assert chat_handler.metadata["output"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_type_chat_invoke(self):
        """type=chat agent handles invoke with messages input."""

        @agent(type="chat")
        async def chat_handler(messages: list):
            last = messages[-1] if messages else {}
            return f"Reply to: {last.get('content', '')}"

        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hi"}]})
        response = await chat_handler.invoke(request)
        assert response["output"] == "Reply to: Hi"

    def test_type_task_metadata(self):
        """type=task sets task input and structured output in metadata."""

        @agent(type="task")
        async def task_handler(task: str, text: str | None = None):
            return f"Task {task}"

        assert task_handler.metadata["type"] == "task"
        assert task_handler.metadata["input"]["required"] == ["task"]
        assert "task" in task_handler.metadata["input"]["properties"]
        assert "description" in task_handler.metadata["output"]
        assert "stateless, single-shot" in task_handler.metadata["output"]["description"]

    @pytest.mark.asyncio
    async def test_type_task_invoke(self):
        """type=task agent handles invoke with task input."""

        @agent(type="task")
        async def task_handler(task: str, text: str | None = None):
            return f'Task "{task}" on: {text or "—"}'

        request = AgentRequest(input={"task": "summarize", "text": "Some content"})
        response = await task_handler.invoke(request)
        assert response["output"] == 'Task "summarize" on: Some content'

    def test_type_workflow_metadata(self):
        """type=workflow sets workflow input/output schemas with status and steps."""

        @agent(type="workflow")
        async def workflow_handler(task: str, steps: list | None = None):
            return {
                "status": "completed",
                "steps": [{"name": "step1", "status": "completed", "output": "done"}],
                "result": {"summary": "All steps completed"},
            }

        assert workflow_handler.metadata["type"] == "workflow"
        input_schema = workflow_handler.metadata["input"]
        assert input_schema["required"] == ["task"]
        assert "task" in input_schema["properties"]
        assert "steps" in input_schema["properties"]
        assert "resume" in input_schema["properties"]
        assert input_schema["additionalProperties"] is True

        output_schema = workflow_handler.metadata["output"]
        assert output_schema["required"] == ["status", "steps"]
        assert "status" in output_schema["properties"]
        assert output_schema["properties"]["status"]["enum"] == [
            "completed",
            "failed",
            "paused",
            "running",
        ]
        assert "steps" in output_schema["properties"]
        assert "result" in output_schema["properties"]
        assert "pendingAction" in output_schema["properties"]

    @pytest.mark.asyncio
    async def test_type_workflow_invoke(self):
        """type=workflow agent handles invoke with task+steps and returns structured output."""

        @agent(type="workflow")
        async def workflow_handler(task: str, steps: list | None = None):
            executed = []
            for s in steps or []:
                executed.append({"name": s["name"], "status": "completed", "output": "ok"})
            return {
                "status": "completed",
                "steps": executed,
                "result": {"summary": f"Ran {len(executed)} steps for: {task}"},
            }

        request = AgentRequest(
            input={
                "task": "process-data",
                "steps": [{"name": "fetch"}, {"name": "transform"}],
            }
        )
        response = await workflow_handler.invoke(request)
        output = response["output"]
        assert output["status"] == "completed"
        assert len(output["steps"]) == 2
        assert output["steps"][0]["name"] == "fetch"
        assert output["steps"][1]["name"] == "transform"
        assert output["result"]["summary"] == "Ran 2 steps for: process-data"

    def test_no_type_derives_from_function(self):
        """Without type, input/output are derived from function signature."""

        @agent
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        assert "type" not in add.metadata
        assert "a" in add.metadata["input"]["properties"]
        assert "b" in add.metadata["input"]["properties"]
        assert add.metadata["input"]["required"] == ["a", "b"]

    def test_type_rag_metadata(self):
        """type=rag sets query input and string output in metadata."""

        @agent(type="rag")
        async def rag_handler(query: str):
            return f"Answer for: {query}"

        assert rag_handler.metadata["type"] == "rag"
        assert rag_handler.metadata["input"]["required"] == ["query"]
        assert "query" in rag_handler.metadata["input"]["properties"]
        assert rag_handler.metadata["output"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_type_rag_invoke(self):
        """type=rag agent handles invoke with query input."""

        @agent(type="rag")
        async def rag_handler(query: str):
            return f"Answer for: {query}"

        request = AgentRequest(input={"query": "What is X?"})
        response = await rag_handler.invoke(request)
        assert response["output"] == "Answer for: What is X?"

    def test_type_thread_metadata(self):
        """type=thread sets messages input and messages output (array) in metadata."""

        @agent(type="thread")
        async def thread_handler(messages: list):
            return messages + [{"role": "assistant", "content": "ok"}]

        assert thread_handler.metadata["type"] == "thread"
        assert thread_handler.metadata["input"]["required"] == ["messages"]
        assert "messages" in thread_handler.metadata["input"]["properties"]
        assert thread_handler.metadata["output"]["type"] == "array"
        assert "items" in thread_handler.metadata["output"]

    @pytest.mark.asyncio
    async def test_type_thread_invoke(self):
        """type=thread agent returns updated message thread (output is messages)."""

        @agent(type="thread")
        async def thread_handler(messages: list):
            last = messages[-1] if messages else {}
            return messages + [
                {"role": "assistant", "content": f"Reply: {last.get('content', '')}"}
            ]

        request = AgentRequest(input={"messages": [{"role": "user", "content": "Hello"}]})
        response = await thread_handler.invoke(request)
        output = response["output"]
        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0] == {"role": "user", "content": "Hello"}
        assert output[1]["role"] == "assistant"
        assert output[1]["content"] == "Reply: Hello"
