"""Tests for BaseAdapter."""

import pytest

from reminix_runtime import BaseAdapter, ChatRequest, ChatResponse, InvokeRequest, InvokeResponse


class TestBaseAdapterContract:
    """Tests for the BaseAdapter contract."""

    def test_cannot_instantiate_base_adapter(self):
        """BaseAdapter is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAdapter()  # type: ignore

    def test_subclass_must_implement_name(self):
        """Subclass must implement the name property."""

        class IncompleteAdapter(BaseAdapter):
            async def invoke(self, request: InvokeRequest) -> InvokeResponse:
                return InvokeResponse(output="")

            async def chat(self, request: ChatRequest) -> ChatResponse:
                return ChatResponse(output="", messages=[])

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_subclass_must_implement_invoke(self):
        """Subclass must implement the invoke method."""

        class IncompleteAdapter(BaseAdapter):
            @property
            def name(self) -> str:
                return "test"

            async def chat(self, request: ChatRequest) -> ChatResponse:
                return ChatResponse(output="", messages=[])

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore

    def test_subclass_must_implement_chat(self):
        """Subclass must implement the chat method."""

        class IncompleteAdapter(BaseAdapter):
            @property
            def name(self) -> str:
                return "test"

            async def invoke(self, request: InvokeRequest) -> InvokeResponse:
                return InvokeResponse(output="")

        with pytest.raises(TypeError):
            IncompleteAdapter()  # type: ignore


class TestConcreteAdapter:
    """Tests for a concrete adapter implementation."""

    def _create_adapter(self) -> BaseAdapter:
        """Create a minimal concrete adapter for testing."""

        class TestAdapter(BaseAdapter):
            @property
            def name(self) -> str:
                return "test-agent"

            async def invoke(self, request: InvokeRequest) -> InvokeResponse:
                task = request.input.get("task", "unknown")
                return InvokeResponse(output=f"Completed: {task}")

            async def chat(self, request: ChatRequest) -> ChatResponse:
                user_msg = request.messages[-1].content
                return ChatResponse(
                    output=f"Hello from chat: {user_msg}",
                    messages=[{"role": "assistant", "content": f"Hello from chat: {user_msg}"}],
                )

        return TestAdapter()

    def test_adapter_has_name(self):
        """Adapter should have a name property."""
        adapter = self._create_adapter()
        assert adapter.name == "test-agent"

    @pytest.mark.asyncio
    async def test_invoke_returns_response(self):
        """Invoke should return an InvokeResponse."""
        adapter = self._create_adapter()
        request = InvokeRequest(input={"task": "summarize"})
        response = await adapter.invoke(request)

        assert isinstance(response, InvokeResponse)
        assert response.output == "Completed: summarize"

    @pytest.mark.asyncio
    async def test_chat_returns_response(self):
        """Chat should return a ChatResponse."""
        adapter = self._create_adapter()
        request = ChatRequest(messages=[{"role": "user", "content": "hello"}])
        response = await adapter.chat(request)

        assert isinstance(response, ChatResponse)
        assert response.output == "Hello from chat: hello"
        assert len(response.messages) == 1

    @pytest.mark.asyncio
    async def test_invoke_stream_not_implemented_by_default(self):
        """invoke_stream should raise NotImplementedError by default."""
        adapter = self._create_adapter()
        request = InvokeRequest(input={"task": "test"})

        with pytest.raises(NotImplementedError):
            async for _ in adapter.invoke_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_chat_stream_not_implemented_by_default(self):
        """chat_stream should raise NotImplementedError by default."""
        adapter = self._create_adapter()
        request = ChatRequest(messages=[{"role": "user", "content": "hello"}])

        with pytest.raises(NotImplementedError):
            async for _ in adapter.chat_stream(request):
                pass
