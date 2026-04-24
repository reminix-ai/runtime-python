# Starter Tool

The fastest way to build a tool on Reminix — a minimal MCP tool with a JSON schema input, no API keys, no external SDKs.

[![Deploy to Reminix](https://reminix.com/badge/deploy.svg)](https://reminix.com/new/deploy?repo=reminix-ai/runtime-python&folder=examples/starter-tool)

## How it works

The `@tool` decorator registers a typed async function as an MCP tool. Input field descriptions come from the `Args:` section of the docstring; the output schema is a Pydantic `BaseModel` with `Field(description=...)` hints. `serve()` hosts everything behind a Streamable HTTP `/mcp` endpoint, so any MCP-compatible client (Claude Desktop, an agent framework, another Reminix agent) can discover and call it using the standard JSON-RPC protocol.

## What it does

```python
from pydantic import BaseModel, Field

from reminix_runtime import serve, tool


class GreetOutput(BaseModel):
    greeting: str = Field(description="Generated greeting")


@tool(name="greet")
async def greet(name: str) -> GreetOutput:
    """Greet someone by name.

    Args:
        name: Name of the person to greet.
    """
    return GreetOutput(greeting=f"Hello, {name}!")


if __name__ == "__main__":
    serve(tools=[greet])
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/manifest` | GET | Tool discovery |
| `/mcp` | POST | MCP Streamable HTTP — tool discovery and invocation |

## Testing

```bash
# List available tools
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/list","id":1}'

# Call the greet tool
curl -X POST http://localhost:8080/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"tools/call","params":{"name":"greet","arguments":{"name":"World"}},"id":2}'
```

## Run locally

```bash
uv sync
uv run python main.py
```
