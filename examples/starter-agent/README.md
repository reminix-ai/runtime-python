# Starter Agent

The fastest way to get started with Reminix — a minimal agent with a single handler, no API keys, no external SDKs.

[![Deploy to Reminix](https://reminix.com/badge/deploy.svg)](https://reminix.com/new/deploy?repo=reminix-ai/runtime-python&folder=examples/starter-agent)

## How it works

Reminix's `@agent` decorator turns a plain async function into a registered agent. The input schema is inferred from the function's type hints and the output schema from its return annotation. `serve()` then exposes it as a REST API with health, manifest, and invoke endpoints — no server code to write.

## What it does

```python
from reminix_runtime import agent, serve


@agent(name="starter-agent")
async def starter(message: str = "") -> str:
    """A minimal starter agent."""
    return f"You said: {message}"


if __name__ == "__main__":
    serve(agents=[starter])
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/manifest` | GET | Agent discovery |
| `/agents/starter-agent/invoke` | POST | Execute the agent |

## Testing

```bash
curl -X POST http://localhost:8080/agents/starter-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}}'

# Response: {"output": "You said: Hello!"}
```

## Run locally

```bash
uv sync
uv run python main.py
```
