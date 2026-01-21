# reminix-runtime

Core runtime package for serving AI agents and tools via REST APIs. Provides the `@agent`, `@chat_agent`, and `@tool` decorators for building and serving AI agents.

Built on [FastAPI](https://fastapi.tiangolo.com) with full async support.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

## Installation

```bash
pip install reminix-runtime
```

## Quick Start

```python
from reminix_runtime import agent, chat_agent, serve, Message

# Create an invoke agent
@agent
async def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

# Create a chat agent
@chat_agent
async def assistant(messages: list[Message]) -> str:
    """A helpful assistant."""
    return f"You said: {messages[-1].content}"

# Serve the agents
serve(agents=[calculator, assistant], port=8080)
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
      "name": "calculator",
      "type": "agent",
      "description": "Add two numbers.",
      "parameters": {
        "type": "object",
        "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
        "required": ["a", "b"]
      },
      "output": { "type": "number" },
      "invoke": { "streaming": false },
      "chat": { "streaming": false }
    },
    {
      "name": "assistant",
      "type": "agent",
      "description": "A helpful assistant.",
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
curl -X POST http://localhost:8080/agents/calculator/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"a": 5, "b": 3}}'
```

**Response:**
```json
{
  "output": 8.0
}
```

### Chat Endpoint

`POST /agents/{name}/chat` - For conversational interactions.

```bash
curl -X POST http://localhost:8080/agents/assistant/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Response:**
```json
{
  "output": "You said: Hello!",
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "You said: Hello!"}
  ]
}
```

### Tool Execute Endpoint

`POST /tools/{name}/execute` - Execute a standalone tool.

```bash
curl -X POST http://localhost:8080/tools/get_weather/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"location": "San Francisco"}}'
```

**Response:**
```json
{
  "output": { "temp": 72, "condition": "sunny" }
}
```

## Agents

Agents handle requests via the `/agents/{name}/invoke` or `/agents/{name}/chat` endpoints.

### Invoke Agent

Use `@agent` for task-oriented agents that take input and return output:

```python
from reminix_runtime import agent, serve

@agent
async def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@agent(name="text-processor", description="Process text in various ways")
async def process_text(text: str, operation: str = "uppercase") -> str:
    """Process text with the specified operation."""
    if operation == "uppercase":
        return text.upper()
    elif operation == "lowercase":
        return text.lower()
    return text

serve(agents=[calculator, process_text], port=8080)
```

The decorator automatically extracts:
- **name** from the function name (or use `name=` to override)
- **description** from the docstring (or use `description=` to override)
- **parameters** from type hints and defaults
- **output** from the return type hint

### Chat Agent

Use `@chat_agent` for conversational agents that handle message history:

```python
from reminix_runtime import chat_agent, serve, Message

@chat_agent
async def assistant(messages: list[Message]) -> str:
    """A helpful assistant."""
    last_msg = messages[-1].content
    return f"You said: {last_msg}"

# With context support
@chat_agent
async def contextual_bot(messages: list[Message], context: dict | None = None) -> str:
    """Bot with context awareness."""
    user_id = context.get("user_id") if context else "unknown"
    return f"Hello user {user_id}!"

serve(agents=[assistant, contextual_bot], port=8080)
```

### Streaming

Both decorators support streaming via async generators. When you use `yield` instead of `return`, the agent automatically supports streaming:

```python
from reminix_runtime import agent, chat_agent, serve, Message

# Streaming invoke agent
@agent
async def streamer(text: str):
    """Stream text word by word."""
    for word in text.split():
        yield word + " "

# Streaming chat agent
@chat_agent
async def streaming_assistant(messages: list[Message]):
    """Stream responses token by token."""
    response = f"You said: {messages[-1].content}"
    for char in response:
        yield char

serve(agents=[streamer, streaming_assistant], port=8080)
```

For streaming agents:
- `stream: true` in the request → chunks are sent via SSE
- `stream: false` in the request → chunks are collected and returned as a single response

## Tools

Tools are standalone functions served via `/tools/{name}/execute`. They're useful for exposing utility functions, external API integrations, or any reusable logic.

### Creating Tools

Use the `@tool` decorator to create tools:

```python
from reminix_runtime import tool, serve

@tool
async def get_weather(location: str, units: str = "celsius") -> dict:
    """Get current weather for a location."""
    # Call weather API...
    return {"temp": 72, "condition": "sunny", "location": location}

@tool
def calculate(expression: str) -> dict:
    """Evaluate a math expression."""
    return {"result": eval(expression)}  # Note: use a safe evaluator in production

serve(tools=[get_weather, calculate], port=8080)
```

The decorator automatically extracts:
- **name** from the function name
- **description** from the docstring
- **parameters** from type hints and defaults
- **output** from the return type hint

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
from reminix_runtime import agent, tool, serve

@agent
async def summarizer(text: str) -> str:
    """Summarize text."""
    return text[:100] + "..."

@tool
def calculate(expression: str) -> dict:
    """Perform basic math operations."""
    return {"result": eval(expression)}

serve(agents=[summarizer], tools=[calculate], port=8080)
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

### `@agent`

Decorator to create an invoke agent from a function.

```python
from reminix_runtime import agent

@agent
async def my_agent(param: str, count: int = 5) -> str:
    """Agent description from docstring."""
    return param * count

# With custom name/description
@agent(name="custom_name", description="Custom description")
async def another_agent(x: int) -> int:
    return x * 2

# Streaming agent
@agent
async def streaming_agent(text: str):
    for word in text.split():
        yield word + " "
```

### `@chat_agent`

Decorator to create a chat agent from a function.

```python
from reminix_runtime import chat_agent, Message

@chat_agent
async def my_chat_agent(messages: list[Message]) -> str:
    """Chat agent description."""
    return f"You said: {messages[-1].content}"

# With context
@chat_agent
async def contextual_agent(messages: list[Message], context: dict | None = None) -> str:
    user_id = context.get("user_id") if context else None
    return f"Hello user {user_id}!"

# Streaming chat agent
@chat_agent
async def streaming_chat(messages: list[Message]):
    for token in ["Hello", " ", "world!"]:
        yield token
```

### `@tool`

Decorator to create a tool from a function.

```python
from reminix_runtime import tool

@tool
async def my_tool(param: str, optional_param: int = 10) -> dict:
    """Tool description from docstring."""
    return {"result": param, "value": optional_param}

# With custom name/description
@tool(name="custom_name", description="Custom description")
def another_tool(x: int) -> int:
    return x * 2
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

## Advanced

### Agent Class

For more control, you can use the `Agent` class directly. This is useful when you need both invoke and chat handlers on the same agent, or want more programmatic control.

```python
from reminix_runtime import Agent, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse, serve

agent = Agent("my-agent", metadata={"version": "1.0"})

@agent.on_invoke
async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
    return InvokeResponse(output="Hello!")

@agent.on_chat
async def handle_chat(request: ChatRequest) -> ChatResponse:
    return ChatResponse(output="Hi!", messages=[...])

# Optional: separate streaming handlers
@agent.on_invoke_stream
async def handle_invoke_stream(request: InvokeRequest):
    yield '{"chunk": "Hello"}'
    yield '{"chunk": " world!"}'

serve(agents=[agent], port=8080)
```

### Tool Class

For programmatic tool creation:

```python
from reminix_runtime import Tool, ToolSchema, ToolExecuteRequest, ToolExecuteResponse, serve

async def execute_handler(request: ToolExecuteRequest) -> ToolExecuteResponse:
    location = request.input.get("location", "unknown")
    return ToolExecuteResponse(output={"temp": 72, "location": location})

my_tool = Tool(
    execute_handler,
    name="get_weather",
    description="Get weather for a location",
)

serve(tools=[my_tool], port=8080)
```

### AgentAdapter

For building framework integrations. See the [framework adapter packages](#framework-adapters) for examples.

```python
from reminix_runtime import AgentAdapter, InvokeRequest, InvokeResponse, ChatRequest, ChatResponse

class MyFrameworkAdapter(AgentAdapter):
    adapter_name = "my-framework"

    def __init__(self, client, name: str = "my-framework"):
        self._client = client
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def invoke(self, request: InvokeRequest) -> InvokeResponse:
        result = await self._client.run(request.input)
        return InvokeResponse(output=result)

    async def chat(self, request: ChatRequest) -> ChatResponse:
        result = await self._client.chat(request.messages)
        return ChatResponse(
            output=result,
            messages=[
                *[{"role": m.role, "content": m.content} for m in request.messages],
                {"role": "assistant", "content": result}
            ]
        )
```

### Serverless Deployment

Use `to_asgi()` for serverless deployments:

```python
# AWS Lambda with Mangum
from mangum import Mangum
from reminix_runtime import agent, InvokeResponse

@agent
async def my_agent(task: str) -> str:
    return f"Completed: {task}"

handler = Mangum(my_agent.to_asgi())
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
