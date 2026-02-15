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

Create a `.env` file in the repository root with your API key:

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
| `/agents/anthropic-basic/invoke` | POST | Execute agent |
| `/agents/anthropic-basic/invoke` | POST | Execute agent |

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
curl -X POST http://localhost:8080/agents/anthropic-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## How it works

1. Create an Anthropic client using `anthropic`
2. Serve it with `reminix-anthropic`

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import serve_agent

client = AsyncAnthropic()

serve_agent(client, name="anthropic-basic", model="claude-3-haiku-20240307")
```
