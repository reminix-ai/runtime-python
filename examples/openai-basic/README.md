# OpenAI Basic Example

A simple example showing how to serve an OpenAI chat agent via Reminix Runtime.

## Setup

```bash
# From the repository root
uv sync

# Navigate to this example
cd examples/openai-basic
```

## Environment

Create a `.env` file in the repository root with your API key:

```bash
OPENAI_API_KEY=your-api-key
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
| `/agents/openai-basic/invoke` | POST | Execute agent |
| `/agents/openai-basic/invoke` | POST | Execute agent |

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery
curl http://localhost:8080/info

# Invoke
curl -X POST http://localhost:8080/agents/openai-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is the capital of France?"}}'

# Chat
curl -X POST http://localhost:8080/agents/openai-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## How it works

1. Create an OpenAI client using `openai`
2. Wrap it with `reminix-openai`
3. Serve it with `reminix-runtime`

```python
from openai import AsyncOpenAI
from reminix_openai import wrap_agent
from reminix_runtime import serve

client = AsyncOpenAI()
agent = wrap_agent(client, name="openai-basic", model="gpt-4o-mini")

serve(agents=[agent], port=8080)
```
