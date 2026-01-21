# Custom Agent Example

This example demonstrates how to create a custom agent using the decorator-based `Agent` class from `reminix-runtime`.

## Running

```bash
cd examples/custom-agent
uv run python main.py
```

## Endpoints

Once running, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Agent discovery |
| `/agents/echo/execute` | POST | Execute the agent |

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery
curl http://localhost:8080/info

# Task-style execution
curl -X POST http://localhost:8080/agents/echo/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}}'

# Chat-style execution
curl -X POST http://localhost:8080/agents/echo/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'

# Streaming execution
curl -X POST http://localhost:8080/agents/echo/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}, "stream": true}'
```
