# LangChain Basic Example

A simple example showing how to serve a LangChain agent via Reminix Runtime.

## Setup

```bash
# From the repository root
uv sync

# Navigate to this example
cd examples/langchain-basic
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
| `/agents/langchain-basic/invoke` | POST | Execute agent |
| `/agents/langchain-basic/invoke` | POST | Execute agent |

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery
curl http://localhost:8080/info

# Invoke
curl -X POST http://localhost:8080/agents/langchain-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is AI?"}}'

# Chat
curl -X POST http://localhost:8080/agents/langchain-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## How it works

1. Create a LangChain chat model using `langchain-openai`
2. Wrap it with `reminix-langchain`
3. Serve it with `reminix-runtime`

```python
from langchain_openai import ChatOpenAI
from reminix_langchain import wrap_agent
from reminix_runtime import serve

model = ChatOpenAI(model="gpt-4o-mini")
agent = wrap_agent(model, name="langchain-basic")

serve(agents=[agent])
```
