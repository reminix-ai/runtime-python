# LlamaIndex RAG Example

An example showing how to serve a LlamaIndex ReAct agent with tools via Reminix Runtime.

## Setup

```bash
# From the repository root
uv sync

# Navigate to this example
cd examples/llamaindex-rag
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
| `/manifest` | GET | Agent discovery |
| `/agents/llamaindex-rag/invoke` | POST | Execute agent |

## Available Tools

- `get_weather(city)`: Get weather for Paris, London, Tokyo, or New York

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery
curl http://localhost:8080/manifest

# Invoke
curl -X POST http://localhost:8080/agents/llamaindex-rag/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"query": "What is the weather in Paris?"}}'

# Chat
curl -X POST http://localhost:8080/agents/llamaindex-rag/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}'
```

## How it works

1. Define tools as Python functions
2. Create a LlamaIndex ReAct agent using the workflow-based API
3. Wrap it with a ChatEngineWrapper for compatibility
4. Create a `LlamaIndexRagAgent` agent and serve it with `reminix-runtime`

```python
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openai import OpenAI
from reminix_llamaindex import LlamaIndexRagAgent
from reminix_runtime import serve

def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "Sunny, 22°C"

llm = OpenAI(model="gpt-4o-mini")
react_agent = ReActAgent(tools=[get_weather], llm=llm)
# Use ChatEngineWrapper to adapt the workflow agent
engine = ChatEngineWrapper(react_agent)
agent = LlamaIndexRagAgent(engine, name="llamaindex-rag")
serve(agents=[agent])
```
