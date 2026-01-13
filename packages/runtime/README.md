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
        # Task-oriented operation
        task = request.input.get("task", "unknown")
        return InvokeResponse(output=f"Completed: {task}")
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        # Conversational interaction
        user_msg = request.messages[-1].content
        response = f"You said: {user_msg}"
        return ChatResponse(
            output=response,
            messages=[
                *[{"role": m.role, "content": m.content} for m in request.messages],
                {"role": "assistant", "content": response}
            ]
        )

# Serve the agent
serve([MyAgent()], port=8080)
```

## How It Works

The runtime creates a REST server with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Runtime discovery (version, agents, endpoints) |
| `/agents/{name}/invoke` | POST | Task-oriented invocation |
| `/agents/{name}/chat` | POST | Multi-turn conversation |

### Discovery Endpoint

`GET /info` returns runtime information and available agents:

```json
{
  "runtime": {
    "name": "reminix-runtime",
    "version": "0.1.0",
    "language": "python",
    "framework": "fastapi"
  },
  "agents": [
    {
      "name": "my-agent",
      "type": "adapter",
      "adapter": "langchain",
      "endpoints": {
        "invoke": "/agents/my-agent/invoke",
        "chat": "/agents/my-agent/chat"
      }
    }
  ]
}
```

### Invoke Endpoint

For task-oriented operations that take arbitrary input and return output.

**Request:**
```json
{
  "input": {
    "task": "summarize",
    "text": "Lorem ipsum..."
  },
  "stream": false,
  "context": {}
}
```

**Response:**
```json
{
  "output": "Summary: ..."
}
```

### Chat Endpoint

For conversational interactions with message history.

**Request:**
```json
{
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What's the weather?"}
  ],
  "stream": false,
  "context": {}
}
```

**Response:**
```json
{
  "output": "The weather is 72°F and sunny!",
  "messages": [
    {"role": "user", "content": "What's the weather?"},
    {"role": "assistant", "content": null, "tool_calls": [...]},
    {"role": "tool", "content": "72°F, sunny", "tool_call_id": "..."},
    {"role": "assistant", "content": "The weather is 72°F and sunny!"}
  ]
}
```

The `output` field contains the final answer, while `messages` includes the full execution history (useful for agentic workflows with tool calls).

## Framework Adapters

Instead of creating custom agents, use our pre-built adapters for popular frameworks:

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
    # Adapter name shown in /info endpoint
    adapter_name = "my-framework"

    def __init__(self, client, name: str = "my-framework"):
        self._client = client
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        # Pass input to your framework
        result = await self._client.run(request.input)
        return InvokeResponse(output=result)
    
    async def chat(self, request: ChatRequest) -> ChatResponse:
        # Convert messages and call your framework
        result = await self._client.chat(request.messages)
        return ChatResponse(
            output=result,
            messages=[
                *[{"role": m.role, "content": m.content} for m in request.messages],
                {"role": "assistant", "content": result}
            ]
        )

# Optional: provide a wrap() factory function
def wrap(client, name: str = "my-framework") -> MyFrameworkAdapter:
    return MyFrameworkAdapter(client, name)
```

### Request/Response Types

```python
class InvokeRequest:
    input: dict[str, Any]      # Arbitrary input for task execution
    stream: bool = False       # Whether to stream the response
    context: dict[str, Any] | None  # Optional metadata

class InvokeResponse:
    output: Any                # The result (can be any type)

class ChatRequest:
    messages: list[Message]    # Conversation history
    stream: bool = False       # Whether to stream the response
    context: dict[str, Any] | None  # Optional metadata

class ChatResponse:
    output: str                # The final answer
    messages: list[dict]       # Full execution history
```

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [Documentation](https://docs.reminix.ai)

## License

Apache-2.0
