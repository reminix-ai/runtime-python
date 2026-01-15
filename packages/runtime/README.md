# reminix-runtime

Core runtime package for serving AI agents via REST APIs. Provides the `serve()` function, `Agent` class, and `BaseAdapter` for building framework integrations.

Built on [FastAPI](https://fastapi.tiangolo.com) with full async support.

## Installation

```bash
pip install reminix-runtime
```

## Quick Start

```python
from reminix_runtime import serve, Agent, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

# Create an agent with decorators
agent = Agent("my-agent")

@agent.on_invoke
async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
    task = request.input.get("task", "unknown")
    return InvokeResponse(output=f"Completed: {task}")

@agent.on_chat
async def handle_chat(request: ChatRequest) -> ChatResponse:
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
serve([agent], port=8080)
```

## How It Works

The runtime creates a REST server with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Runtime discovery (version, agents, endpoints) |
| `/agents/{name}/invoke` | POST | Stateless invocation |
| `/agents/{name}/chat` | POST | Conversational chat |

### Health Endpoint

```bash
curl http://localhost:8080/health
```

Returns `{"status": "ok"}` if the server is running.

### Discovery Endpoint

```bash
curl http://localhost:8080/info
```

Returns runtime information and available agents:

```json
{
  "runtime": {
    "name": "reminix-runtime",
    "version": "0.0.1",
    "language": "python",
    "framework": "fastapi"
  },
  "agents": [
    {
      "name": "my-agent",
      "type": "adapter",
      "adapter": "langchain",
      "invoke": { "streaming": true },
      "chat": { "streaming": true }
    }
  ]
}
```

### Invoke Endpoint

`POST /agents/{name}/invoke` - For stateless operations.

```bash
curl -X POST http://localhost:8080/agents/my-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "task": "summarize",
      "text": "Lorem ipsum..."
    }
  }'
```

**Response:**
```json
{
  "output": "Summary: ..."
}
```

### Chat Endpoint

`POST /agents/{name}/chat` - For conversational interactions.

```bash
curl -X POST http://localhost:8080/agents/my-agent/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are helpful"},
      {"role": "user", "content": "What is the weather?"}
    ]
  }'
```

**Response:**
```json
{
  "output": "The weather is 72°F and sunny!",
  "messages": [
    {"role": "system", "content": "You are helpful"},
    {"role": "user", "content": "What is the weather?"},
    {"role": "assistant", "content": "The weather is 72°F and sunny!"}
  ]
}
```

The `output` field contains the assistant's response, while `messages` includes the full conversation history.

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

Concrete class for building agents with decorators.

```python
from reminix_runtime import Agent, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

agent = Agent("my-agent", metadata={"version": "1.0"})

@agent.on_invoke
async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
    return InvokeResponse(output="Hello!")

@agent.on_chat
async def handle_chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(output="Hi!", messages=[...])

# Optional: streaming handlers
@agent.on_invoke_stream
async def handle_invoke_stream(request: InvokeRequest):
    yield '{"chunk": "Hello"}'
    yield '{"chunk": " world!"}'

@agent.on_chat_stream
async def handle_chat_stream(request: ChatRequest):
    yield '{"chunk": "Hi"}'
```

| Method | Description |
|--------|-------------|
| `on_invoke(fn)` | Register invoke handler |
| `on_chat(fn)` | Register chat handler |
| `on_invoke_stream(fn)` | Register streaming invoke handler |
| `on_chat_stream(fn)` | Register streaming chat handler |
| `to_asgi()` | Returns an ASGI app for serverless |

### `agent.to_asgi()`

Returns an ASGI application for serverless deployments.

```python
# AWS Lambda with Mangum
from mangum import Mangum
from reminix_runtime import Agent, InvokeResponse

agent = Agent("my-agent")

@agent.on_invoke
async def handle(request):
    return InvokeResponse(output="Hello!")

# Lambda handler
handler = Mangum(agent.to_asgi())
```

Works with:
- **AWS Lambda** - Use Mangum adapter
- **GCP Cloud Functions** - Use functions-framework with ASGI
- **Any ASGI server** - uvicorn, hypercorn, daphne

### `BaseAdapter`

Abstract base class for framework adapters. Use this when wrapping an existing AI framework.

```python
from reminix_runtime import BaseAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

class MyFrameworkAdapter(BaseAdapter):
    # Adapter name shown in /info endpoint
    adapter_name = "my-framework"
    
    # BaseAdapter defaults both to True; override if your adapter doesn't support streaming
    # invoke_streaming = False
    # chat_streaming = False

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
