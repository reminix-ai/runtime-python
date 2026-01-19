# reminix-runtime

Core runtime package for serving AI agents and tools via REST APIs. Provides the `serve()` function, `Agent` class, `@tool` decorator, and `AgentAdapter` for building framework integrations.

Built on [FastAPI](https://fastapi.tiangolo.com) with full async support.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

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
serve(agents=[agent], port=8080)
```

## How It Works

The runtime creates a REST server with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Runtime discovery (version, agents, tools) |
| `/agents/{name}/invoke` | POST | Stateless invocation |
| `/agents/{name}/chat` | POST | Conversational chat |
| `/tools/{name}/execute` | POST | Execute a tool |

### Health Endpoint

```bash
curl http://localhost:8080/health
```

Returns `{"status": "ok"}` if the server is running.

### Discovery Endpoint

```bash
curl http://localhost:8080/info
```

Returns runtime information, available agents, and tools:

```json
{
  "runtime": {
    "name": "reminix-runtime",
    "version": "0.0.7",
    "language": "python",
    "framework": "fastapi"
  },
  "agents": [
    {
      "name": "my-agent",
      "type": "agent",
      "invoke": { "streaming": false },
      "chat": { "streaming": false }
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "type": "tool",
      "description": "Get current weather for a location",
      "parameters": { ... },
      "output": { ... }
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

### Tool Execute Endpoint

`POST /tools/{name}/execute` - Execute a standalone tool.

```bash
curl -X POST http://localhost:8080/tools/get_weather/execute \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "location": "San Francisco"
    }
  }'
```

**Response:**
```json
{
  "output": { "temp": 72, "condition": "sunny" }
}
```

## Tools

Tools are standalone functions that can be served via the runtime. They're useful for exposing utility functions, external API integrations, or any reusable logic.

### Creating Tools

Use the `@tool` decorator to create tools:

```python
from reminix_runtime import tool, serve

@tool
async def get_weather(location: str, units: str = "celsius") -> dict:
    """Get current weather for a location."""
    # Call weather API...
    return {"temp": 72, "condition": "sunny", "location": location}

# Serve tools (with or without agents)
serve(tools=[get_weather], port=8080)
```

The decorator automatically extracts:
- **name** from the function name
- **description** from the docstring
- **parameters** from type hints and defaults
- **output** from the return type hint (e.g., `-> dict`, `-> str`, `-> list`)

The output schema is included in the `/info` endpoint for documentation and enables better type inference for clients.

### Custom Tool Configuration

You can customize the tool name and description:

```python
@tool(name="weather_lookup", description="Look up weather for any city")
async def get_weather(location: str) -> dict:
    return {"temp": 72, "condition": "sunny"}
```

### Serving Agents and Tools Together

You can serve both agents and tools from the same runtime:

```python
from reminix_runtime import Agent, tool, serve

agent = Agent("my-agent")

@agent.on_invoke
async def handle(request):
    return {"output": "Hello!"}

@tool
def calculate(expression: str) -> dict:
    """Perform basic math operations."""
    return {"result": eval(expression)}  # Note: use a safe evaluator in production

serve(agents=[agent], tools=[calculate], port=8080)
```

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

### `serve(agents, tools, port, host)`

Start the runtime server.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `list[Agent]` | `[]` | List of agents to serve |
| `tools` | `list[Tool]` | `[]` | List of tools to serve |
| `port` | `int` | `8080` | Port to listen on. Falls back to `PORT` environment variable if not provided. |
| `host` | `str` | `"0.0.0.0"` | Host to bind to (all interfaces). Can be overridden via `HOST` env var. |

At least one agent or tool is required.

### `create_app(agents, tools)`

Create a FastAPI app without starting the server. Useful for testing or custom deployment.

```python
from reminix_runtime import create_app

app = create_app(agents=[my_agent], tools=[my_tool])
# Use with uvicorn, gunicorn, etc.
```

### `@tool`

Decorator to create a tool from a function.

```python
from reminix_runtime import tool

@tool
async def my_tool(param: str, optional_param: int = 10) -> dict:
    """Tool description from docstring."""
    return {"result": param, "value": optional_param}

# Or with custom name/description
@tool(name="custom_name", description="Custom description")
def another_tool(x: int) -> int:
    return x * 2
```

The decorator automatically extracts parameters from type hints. Supports both sync and async functions.

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

### `AgentAdapter`

Abstract base class for framework adapters. Use this when wrapping an existing AI framework.

```python
from reminix_runtime import AgentAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

class MyFrameworkAdapter(AgentAdapter):
    # Adapter name shown in /info endpoint
    adapter_name = "my-framework"
    
    # AgentAdapter defaults both to True; override if your adapter doesn't support streaming
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

## Deployment

Ready to go live?

- **[Deploy to Reminix Cloud](https://reminix.com/docs/deployment)** - Zero-config cloud hosting
- **[Self-host](https://reminix.com/docs/deployment/self-hosting)** - Run on your own infrastructure

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [Documentation](https://reminix.com/docs)

## License

Apache-2.0
