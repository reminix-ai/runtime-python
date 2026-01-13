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
| `/agents/echo/invoke` | POST | Task-oriented invocation |
| `/agents/echo/chat` | POST | Conversational chat |

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery
curl http://localhost:8080/info

# Invoke
curl -X POST http://localhost:8080/agents/echo/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}}'

# Chat
curl -X POST http://localhost:8080/agents/echo/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming
curl -X POST http://localhost:8080/agents/echo/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"message": "Hello!"}, "stream": true}'
```
