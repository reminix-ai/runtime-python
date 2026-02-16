# LangGraph with Tools Example

An example showing how to serve a LangGraph ReAct agent with tool calling via Reminix Runtime.

## Setup

```bash
# From the repository root
uv sync

# Navigate to this example
cd examples/langgraph-tools
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
| `/agents/langgraph-tools/invoke` | POST | Execute agent |

## Available Tools

- `get_weather(city)`: Get weather for Paris, London, Tokyo, or New York

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery
curl http://localhost:8080/manifest

# Invoke (with tool use)
curl -X POST http://localhost:8080/agents/langgraph-tools/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]}}'

# Chat
curl -X POST http://localhost:8080/agents/langgraph-tools/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}'
```

## How it works

1. Define tools using `@tool` decorator from LangChain
2. Create a LangGraph ReAct agent using `create_react_agent`
3. Create a `LangGraphThreadAgent` agent and serve it with `reminix-runtime`

```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from reminix_langgraph import LangGraphThreadAgent
from reminix_runtime import serve

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return "Sunny, 22°C"

llm = ChatOpenAI(model="gpt-4o-mini")
graph = create_react_agent(llm, tools=[get_weather])
agent = LangGraphThreadAgent(graph, name="langgraph-tools")
serve(agents=[agent])
```
