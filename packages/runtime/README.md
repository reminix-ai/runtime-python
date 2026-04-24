# reminix-runtime

The open source runtime for serving AI agents via REST APIs. Part of [Reminix](https://reminix.com) — the developer platform for AI agents.

Core runtime package for serving AI agents and tools via REST APIs. Provides the `@agent` and `@tool` decorators, agent types (prompt, chat, task, thread, workflow), and types `Message` and `ToolCall` for OpenAI-style conversations.

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
serve(agents=[calculator])
```

## How It Works

The runtime creates a REST server with the following endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/manifest` | GET | Runtime discovery (version, endpoints) |
| `/agents/{name}/invoke` | POST | Invoke an agent |
| `/mcp` | POST | MCP Streamable HTTP (tool discovery and execution) |

### Health Endpoint

```bash
curl http://localhost:8080/health
```

Returns `{"status": "ok"}` if the server is running.

### Discovery Endpoint

```bash
curl http://localhost:8080/manifest
```

Returns runtime information and available endpoints:

```json
{
  "runtime": {
    "name": "reminix-runtime",
    "version": "0.0.22",
    "language": "python"
  },
  "endpoints": [
    {
      "kind": "agent",
      "path": "/agents/calculator/invoke",
      "name": "calculator",
      "description": "Add two numbers.",
      "capabilities": { "streaming": false },
      "inputSchema": {
        "type": "object",
        "properties": { "a": { "type": "number" }, "b": { "type": "number" } },
        "required": ["a", "b"]
      },
      "outputSchema": { "type": "number" }
    },
    {
      "kind": "mcp",
      "path": "/mcp"
    }
  ]
}
```

### Agent Invoke Endpoint

`POST /agents/{name}/invoke` - Invoke an agent.

Request body contains an `input` object with keys defined by the agent's input schema, plus optional top-level `stream` and `context` fields. For example, a calculator agent with input schema `{ properties: { a, b } }` expects `a` and `b` inside the `input` object:

**Task-oriented agent:**
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

**Chat agent:**

Chat agents (type `chat` or `thread`) expect `messages` inside the `input` object. Messages are OpenAI-style: `role` (`user` | `assistant` | `system` | `tool`), `content`, and optionally `tool_calls`, `tool_call_id`, and `name`. Use the `Message` and `ToolCall` types from `reminix_runtime` in your handler. Chat returns a string; thread returns a list of messages.

```bash
curl -X POST http://localhost:8080/agents/assistant/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"role": "user", "content": "Hello!"}
      ]
    }
  }'
```

**Response (chat):**
```json
{
  "output": "You said: Hello!"
}
```

### MCP Endpoint

`POST /mcp` - MCP Streamable HTTP endpoint for tool discovery and execution.

Tools are exposed via [MCP (Model Context Protocol)](https://modelcontextprotocol.io) at `/mcp`. Use any MCP client, or call directly with JSON-RPC:

```bash
# Discover available tools
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

# Call a tool
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "get_weather", "arguments": {"location": "San Francisco"}}, "id": 2}'
```

## Agents

Agents handle requests via the `/agents/{name}/invoke` endpoint.

### Agent types

You can use a **type** to get standard input/output schemas without defining them yourself. Pass `type` to the `@agent` decorator:

| Type | Input | Output | Use case |
|----------|--------|--------|----------|
| `prompt` (default) | `{ prompt: str }` | `str` | Single prompt in, text out |
| `chat` | `{ messages: list[Message] }` | `str` | Multi-turn chat, final reply as string |
| `task` | `{ task: str, ... }` | JSON | Stateless, single-shot execution with structured result |
| `thread` | `{ messages: list[Message] }` | `list[Message]` | Multi-turn with tool calls; returns updated thread |
| `workflow` | `{ task: str, steps?: list, resume?: object, ... }` | `{ status, steps, result?, pendingAction? }` | Multi-step orchestration with branching, approvals, and parallel execution |

Messages are OpenAI-style: `role`, `content`, and optionally `tool_calls`, `tool_call_id`, and `name`. Use the exported types `Message` and `ToolCall` from `reminix_runtime` for type-safe handlers. `Message.tool_calls` is `list[ToolCall] | None`.

```python
from reminix_runtime import agent, serve, Message, ToolCall

@agent(type="chat", description="Helpful assistant")
async def assistant(messages: list[Message]) -> str:
    last = messages[-1] if messages else None
    return f"You said: {last.content}" if last and last.role == "user" else "Hello!"

serve(agents=[assistant])
```

### Task-Oriented Agent

Use `@agent` for task-oriented agents that take structured input and return output (omit `type` or use `type="prompt"` or `type="task"` for standard shapes):

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

serve(agents=[calculator, process_text])
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

serve(agents=[streamer])
```

For streaming agents:
- `stream: true` in the request → chunks are sent via SSE
- `stream: false` in the request → chunks are collected and returned as a single response

## Tools

Tools are standalone functions exposed via [MCP](https://modelcontextprotocol.io) at `/mcp`. They're useful for exposing utility functions, external API integrations, or any reusable logic. MCP clients (including LLMs and other agents) can discover and call tools using the standard MCP protocol.

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

serve(tools=[get_weather])
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

serve(agents=[summarizer], tools=[calculate])
```

## Framework Agents

Already using a framework? Use our pre-built agents:

| Package | Framework |
|---------|-----------|
| [`reminix-langchain`](https://pypi.org/project/reminix-langchain/) | LangChain |
| [`reminix-langgraph`](https://pypi.org/project/reminix-langgraph/) | LangGraph |
| [`reminix-openai`](https://pypi.org/project/reminix-openai/) | OpenAI |
| [`reminix-anthropic`](https://pypi.org/project/reminix-anthropic/) | Anthropic |

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

Decorator to create an agent from a function. Use `type` for standard I/O shapes, or let the decorator infer input/output from type hints.

| Parameter | Description |
|-----------|-------------|
| `type` | `"prompt"` \| `"chat"` \| `"task"` \| `"thread"` \| `"workflow"`. Standard input/output schema (default: `"prompt"` when no custom input/output). |
| `name` | Agent name (default: function name) |
| `description` | Human-readable description (default: from docstring) |

```python
from reminix_runtime import agent

@agent
async def my_agent(param: str, count: int = 5) -> str:
    """Agent description from docstring."""
    return param * count

# With type (e.g. chat)
@agent(type="chat", description="Helpful assistant")
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

To receive request context (e.g. `user_id` from the request body), add an optional `context` parameter: `async def my_agent(param: str, context: dict | None = None) -> str:`.

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

# With context (optional parameter receives request context)
@tool
async def my_tool(param: str, context: dict | None = None) -> dict:
    return {"param": param, "user_id": (context or {}).get("user_id", "anonymous")}
```

### Request/Response Types

```python
# Request: { "input": { ... }, "stream": bool, "context": { ... } }
# For a calculator agent with input schema { a: float, b: float }:
# {
#   "input": {
#     "a": 5,                       # From input schema
#     "b": 3                        # From input schema
#   },
#   "stream": false,                # Whether to stream the response
#   "context": { ... }              # Optional metadata
# }

# For a chat agent:
# {
#   "input": {
#     "messages": [...]             # From input schema
#   },
#   "stream": false,
#   "context": { ... }
# }

# Response: { "output": ... } (value from handler)
# Chat type: output is a string (final reply)
# Thread type: output is a list of Message (updated thread)
```

## Advanced

### Agent Base Class

For building framework integrations, extend the `Agent` base class. See the [framework agent packages](#framework-agents) for examples.

```python
from reminix_runtime import Agent, AgentRequest, serve

class MyFrameworkAgent(Agent):
    """Wraps a framework client as an Agent."""

    def __init__(self, client, name: str = "my-framework"):
        super().__init__(name=name, description="My framework agent")
        self._client = client

    async def invoke(self, request: AgentRequest) -> dict:
        result = await self._client.run(request.input)
        return {"output": result}

serve(agents=[MyFrameworkAgent(client)])
```

### Tool Factory

For tools with explicit schemas, use the `tool()` factory with keyword arguments:

```python
from reminix_runtime import tool, serve

@tool(name="get_weather", description="Get weather for a location")
async def get_weather(location: str, units: str = "celsius") -> dict:
    """Get current weather.

    Args:
        location: City name
        units: Temperature units
    """
    return {"temp": 72, "location": location, "units": units}

serve(tools=[get_weather])
```

### Serverless Deployment

Use `create_app()` for serverless deployments:

```python
# AWS Lambda with Mangum
from mangum import Mangum
from reminix_runtime import agent, create_app

@agent
async def my_agent(task: str) -> str:
    return f"Completed: {task}"

app = create_app(agents=[my_agent])
handler = Mangum(app)
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
