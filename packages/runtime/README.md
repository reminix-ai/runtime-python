# reminix-runtime

The core runtime for deploying AI agents via REST APIs. Provides a lightweight server with a unified interface for any AI framework.

## Installation

```bash
pip install reminix-runtime
```

## Quick Start

```python
from reminix_runtime import serve, BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

# Create a custom adapter
class MyAdapter(BaseAdapter):
    @property
    def name(self) -> str:
        return "my-agent"
    
    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        # Your agent logic here
        return InvokeResponse(
            content="Hello!",
            messages=[*request.messages, {"role": "assistant", "content": "Hello!"}]
        )
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        return ChatResponse(
            content="Hello!",
            messages=[*request.messages, {"role": "assistant", "content": "Hello!"}]
        )

# Serve the agent
serve([MyAdapter()], port=8080)
```

## How It Works

The runtime creates a REST server with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/agents` | GET | List available agents |
| `/{agent}/invoke` | POST | Single-turn invocation |
| `/{agent}/chat` | POST | Multi-turn chat |

### Request Format

```json
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"}
  ],
  "context": {}
}
```

### Response Format

```json
{
  "content": "Hi there! How can I help?",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help?"}
  ]
}
```

## Framework Adapters

Instead of creating custom adapters, use our pre-built adapters for popular frameworks:

| Package | Framework |
|---------|-----------|
| [`reminix-langchain`](https://pypi.org/project/reminix-langchain/) | LangChain |
| [`reminix-langgraph`](https://pypi.org/project/reminix-langgraph/) | LangGraph |
| [`reminix-openai`](https://pypi.org/project/reminix-openai/) | OpenAI |
| [`reminix-anthropic`](https://pypi.org/project/reminix-anthropic/) | Anthropic |
| [`reminix-llamaindex`](https://pypi.org/project/reminix-llamaindex/) | LlamaIndex |

## API Reference

### `serve(agents, port, host)`

Start the runtime server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[BaseAdapter]` | required | List of wrapped agents |
| `port` | `int` | `8080` | Port to listen on |
| `host` | `str` | `"0.0.0.0"` | Host to bind to |

### `create_app(agents)`

Create a FastAPI app without starting the server. Useful for testing or custom deployment.

```python
from reminix_runtime import create_app

app = create_app([MyAdapter()])
# Use with uvicorn, gunicorn, etc.
```

### `BaseAdapter`

Abstract base class for all adapters.

```python
class BaseAdapter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    async def invoke(self, request: InvokeRequest) -> InvokeResponse: ...
    
    @abstractmethod
    async def chat(self, request: ChatRequest) -> ChatResponse: ...
    
    # Optional streaming methods
    async def invoke_stream(self, request: InvokeRequest) -> AsyncIterator[str]: ...
    async def chat_stream(self, request: ChatRequest) -> AsyncIterator[str]: ...
```

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [Documentation](https://docs.reminix.ai)

## License

Apache-2.0
