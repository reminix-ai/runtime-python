# reminix-runtime

The open source runtime for serving AI agents via REST APIs. Part of [Reminix](https://reminix.com) — the developer platform for AI agents.

Core runtime package for serving AI agents and tools via REST APIs. Provides the `@agent` and `@tool` decorators, agent templates (prompt, chat, task, rag, thread), and types `Message` and `ToolCall` for OpenAI-style conversations.

Built on [FastAPI](https://fastapi.tiangolo.com) with full async support.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

## Installation

```bash
pip install reminix-runtime
```

## Quick Start

```python
from reminix_runtime import agent, serve

# Create an agent for task-oriented operations
@agent
async def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

# Serve the agent
serve(agents=[calculator], port=8080)
```

## How It Works

The runtime creates a REST server with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Runtime discovery (version, agents, tools) |
| `/agents/{name}/invoke` | POST | Invoke an agent |
| `/tools/{name}/call` | POST | Call a tool |

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
    "version": "0.0.17",
    "language": "python",
    "framework": "fastapi"
  },
  "agents": [
    {
      "name": "calculator",
      "type": "agent",
      "description": "Add two numbers.",
      "input": {
        "type": "object",
        "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
        "required": ["a", "b"]
      },
      "output": {
        "type": "object",
        "properties": { "content": { "type": "number" } },
        "required": ["content"]
      },
      "requestKeys": ["a", "b"],
      "responseKeys": ["content"],
      "streaming": false
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "type": "tool",
      "description": "Get current weather for a location",
      "input": { ... },
      "output": { ... }
    }
  ]
}
```

### Agent Invoke Endpoint

`POST /agents/{name}/invoke` - Invoke an agent.

Request keys are defined by the agent's input schema. For example, a calculator agent with input schema `{ properties: { a, b } }` expects `a` and `b` at the top level:

**Task-oriented agent:**
```bash
curl -X POST http://localhost:8080/agents/calculator/invoke \
  -H "Content-Type: application/json" \
  -d '{"a": 5, "b": 3}'
```

**Response:**
```json
{
  "content": 8.0
}
```

**Chat agent:**

Chat agents (template `chat` or `thread`) expect `messages` at the top level. Messages are OpenAI-style: `role` (`user` | `assistant` | `system` | `tool`), `content`, and optionally `tool_calls`, `tool_call_id`, and `name`. Use the `Message` and `ToolCall` types from `reminix_runtime` in your handler. Chat returns a string; thread returns a list of messages.

```bash
curl -X POST http://localhost:8080/agents/assistant/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**Response (chat):**
```json
{
  "content": "You said: Hello!"
}
```

### Tool Call Endpoint

`POST /tools/{name}/call` - Call a standalone tool.

```bash
curl -X POST http://localhost:8080/tools/get_weather/call \
  -H "Content-Type: application/json" \
  -d '{"location": "San Francisco"}'
```

**Response:**
```json
{
  "content": { "temp": 72, "condition": "sunny" }
}
```

## Agents

Agents handle requests via the `/agents/{name}/invoke` endpoint.

### Agent templates

You can use a **template** to get standard input/output schemas without defining them yourself. Pass `template` to the `@agent` decorator:

| Template | Input | Output | Use case |
|----------|--------|--------|----------|
| `prompt` (default) | `{ prompt: str }` | `str` | Single prompt in, text out |
| `chat` | `{ messages: list[Message] }` | `str` | Multi-turn chat, final reply as string |
| `task` | `{ task: str, ... }` | JSON | Task name + params, structured result |
| `rag` | `{ query: str, messages?: list[Message], collectionIds?: list[str] }` | `str` | RAG query, optional history and collections |
| `thread` | `{ messages: list[Message] }` | `list[Message]` | Multi-turn with tool calls; returns updated thread |

Messages are OpenAI-style: `role`, `content`, and optionally `tool_calls`, `tool_call_id`, and `name`. Use the exported types `Message` and `ToolCall` from `reminix_runtime` for type-safe handlers. `Message.tool_calls` is `list[ToolCall] | None`.

```python
from reminix_runtime import agent, serve, Message, ToolCall

@agent(template="chat", description="Helpful assistant")
async def assistant(messages: list[Message]) -> str:
    last = messages[-1] if messages else None
    return f"You said: {last.content}" if last and last.role == "user" else "Hello!"

serve(agents=[assistant], port=8080)
```

### Task-Oriented Agent

Use `@agent` for task-oriented agents that take structured input and return output (omit `template` or use `template="prompt"` or `template="task"` for standard shapes):

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
- **input schema** from type hints and defaults
- **output** from the return type hint

### Streaming

Agents support streaming via async generators. When you use `yield` instead of `return`, the agent automatically supports streaming:

```python
from reminix_runtime import agent, serve

@agent
async def streamer(text: str):
    """Stream text word by word."""
    for word in text.split():
        yield word + " "

serve(agents=[streamer], port=8080)
```

For streaming agents:
- `stream: true` in the request → chunks are sent via SSE
- `stream: false` in the request → chunks are collected and returned as a single response

## Tools

Tools are standalone functions served via `/tools/{name}/call`. They're useful for exposing utility functions, external API integrations, or any reusable logic.

### Creating Tools

Use the `@tool` decorator to create tools:

```python
from pydantic import BaseModel, Field
from reminix_runtime import tool, serve


class WeatherOutput(BaseModel):
    """Output schema for weather tool."""
    temp: int = Field(description="Temperature value")
    condition: str = Field(description="Weather condition")
    location: str = Field(description="Location name")


@tool
async def get_weather(location: str, units: str = "celsius") -> WeatherOutput:
    """Get current weather for a location.

    Args:
        location: City name to look up
        units: Temperature units (celsius or fahrenheit)
    """
    # Call weather API...
    return WeatherOutput(temp=72, condition="sunny", location=location)

serve(tools=[get_weather], port=8080)
```

The decorator automatically extracts:
- **name** from the function name
- **description** from the docstring (first line/paragraph)
- **input schema** from type hints and defaults
- **input field descriptions** from docstring `Args:` section (Google, NumPy, or Sphinx style)
- **output** from the return type hint (supports Pydantic models, TypedDict, and basic types)

### Output Schema Options

For rich output schemas, use **Pydantic models** (recommended) or **TypedDict**:

```python
from typing import TypedDict
from pydantic import BaseModel, Field

# Option 1: Pydantic (recommended) - includes descriptions and validation
class GreetOutput(BaseModel):
    message: str = Field(description="The greeting message")

@tool
def greet(name: str) -> GreetOutput:
    return GreetOutput(message=f"Hello, {name}!")

# Option 2: TypedDict - simpler, no validation
class CalcOutput(TypedDict):
    result: float

@tool
def calculate(expression: str) -> CalcOutput:
    return {"result": eval(expression)}

# Option 3: Basic types - for simple returns
@tool
def echo(text: str) -> str:
    return text
```

> **Note:** Using `-> dict` loses property information. Use Pydantic or TypedDict for rich schemas.

### Custom Tool Configuration

You can customize the tool name and description:

```python
@tool(name="weather_lookup", description="Look up weather for any city")
async def get_weather(location: str) -> WeatherOutput:
    return WeatherOutput(temp=72, condition="sunny", location=location)
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

Already using a framework? Use our pre-built adapters:

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

Decorator to create an agent from a function. Use `template` for standard I/O shapes, or let the decorator infer input/output from type hints.

| Parameter | Description |
|-----------|-------------|
| `template` | `"prompt"` \| `"chat"` \| `"task"` \| `"rag"` \| `"thread"`. Standard input/output schema (default: `"prompt"` when no custom input/output). |
| `name` | Agent name (default: function name) |
| `description` | Human-readable description (default: from docstring) |

```python
from reminix_runtime import agent

@agent
async def my_agent(param: str, count: int = 5) -> str:
    """Agent description from docstring."""
    return param * count

# With template (e.g. chat)
@agent(template="chat", description="Helpful assistant")
async def assistant(messages: list) -> str:
    return "Hello!"

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

### `@tool`

Decorator to create a tool from a function.

```python
from pydantic import BaseModel, Field
from reminix_runtime import tool


class MyOutput(BaseModel):
    result: str = Field(description="The result")
    value: int = Field(description="The computed value")


@tool
async def my_tool(param: str, optional_param: int = 10) -> MyOutput:
    """Tool description from docstring.

    Args:
        param: The input value
        optional_param: An optional value
    """
    return MyOutput(result=param, value=optional_param)

# With custom name/description
@tool(name="custom_name", description="Custom description")
def another_tool(x: int) -> int:
    return x * 2
```

### Request/Response Types

```python
# Request: top-level keys based on agent's requestKeys (derived from input schema)
# For a calculator agent with input schema { a: float, b: float }:
# {
#   "a": 5,                         # Top-level key from input schema
#   "b": 3,                         # Top-level key from input schema  
#   "stream": false,                # Whether to stream the response
#   "context": { ... }              # Optional metadata
# }

# For a chat agent:
# {
#   "messages": [...],              # Top-level key (requestKeys: ['messages'])
#   "stream": false,
#   "context": { ... }
# }

# Response: { "output": ... } (value from handler)
# Chat template: output is a string (final reply)
# Thread template: output is a list of Message (updated thread)
```

## Advanced

### Agent Class

For more control, you can use the `Agent` class directly:

```python
from reminix_runtime import Agent, ExecuteRequest, ExecuteResponse, serve

agent = Agent("my-agent", metadata={"version": "1.0"})

@agent.handler
async def handle_execute(request: ExecuteRequest) -> ExecuteResponse:
    return ExecuteResponse(output="Hello!")

# Optional: streaming handler
@agent.stream_handler
async def handle_execute_stream(request: ExecuteRequest):
    yield "Hello"
    yield " world!"

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
from reminix_runtime import AgentAdapter, ExecuteRequest, ExecuteResponse

class MyFrameworkAdapter(AgentAdapter):
    adapter_name = "my-framework"

    def __init__(self, client, name: str = "my-framework"):
        self._client = client
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        result = await self._client.run(request.input)
        return ExecuteResponse(output=result)
```

### Serverless Deployment

Use `to_asgi()` for serverless deployments:

```python
# AWS Lambda with Mangum
from mangum import Mangum
from reminix_runtime import agent, ExecuteResponse

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
