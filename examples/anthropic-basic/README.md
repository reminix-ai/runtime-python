# Anthropic Basic Example

A simple example showing how to serve an Anthropic Claude agent via Reminix Runtime.

## Setup

```bash
# From the repository root
uv sync

# Navigate to this example
cd examples/anthropic-basic
```

## Environment

Create a `.env` file with your API key:

```bash
ANTHROPIC_API_KEY=your-api-key
```

## Usage

```bash
uv run python main.py
```

## Endpoints

Once running, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Agent discovery |
| `/agents/anthropic-basic/invoke` | POST | Stateless invocation |
| `/agents/anthropic-basic/chat` | POST | Conversational chat |

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery
curl http://localhost:8080/info

# Invoke
curl -X POST http://localhost:8080/agents/anthropic-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is the capital of France?"}}'

# Chat
curl -X POST http://localhost:8080/agents/anthropic-basic/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## How it works

1. Create an Anthropic client using `anthropic`
2. Wrap it with `reminix-anthropic`
3. Serve it with `reminix-runtime`

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import wrap
from reminix_runtime import serve

client = AsyncAnthropic()
agent = wrap(client, name="anthropic-basic", model="claude-3-haiku-20240307")

serve([agent], port=8080)
```
