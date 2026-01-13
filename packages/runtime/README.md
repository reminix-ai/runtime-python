# reminix-runtime

The core runtime for deploying AI agents via REST APIs. Provides a lightweight server with a unified interface for any AI framework.

## Installation

```bash
pip install reminix-runtime
```

## Quick Start

```python
from reminix_runtime import serve, Agent, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

# Create a custom agent
class MyAgent(Agent):
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
serve([MyAgent()], port=8080)
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
| `agents` | `list[Agent]` | required | List of agents |
| `port` | `int` | `8080` | Port to listen on |
| `host` | `str` | `"0.0.0.0"` | Host to bind to |

### `create_app(agents)`

Create a FastAPI app without starting the server. Useful for testing or custom deployment.

```python
from reminix_runtime import create_app

app = create_app([MyAgent()])
# Use with uvicorn, gunicorn, etc.
```

### `Agent`

Abstract base class for building agents from scratch.

```python
class Agent(ABC):
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

### `BaseAdapter`

Extends `Agent`. Use this when wrapping an existing AI framework.

```python
from reminix_runtime import BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

class MyFrameworkAdapter(BaseAdapter):
    def __init__(self, client, name: str = "my-framework"):
        self._client = client
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        # Convert messages and call your framework
        result = await self._client.generate(request.messages)
        return InvokeResponse(
            content=result,
            messages=[*request.messages, {"role": "assistant", "content": result}]
        )
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        result = await self._client.generate(request.messages)
        return ChatResponse(
            content=result,
            messages=[*request.messages, {"role": "assistant", "content": result}]
        )

# Optional: provide a wrap() factory function
def wrap(client, name: str = "my-framework") -> MyFrameworkAdapter:
    return MyFrameworkAdapter(client, name)
```

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [Documentation](https://docs.reminix.ai)

## License

Apache-2.0
