# reminix-langgraph

Reminix Runtime adapter for [LangGraph](https://langchain-ai.github.io/langgraph/). Deploy any LangGraph agent as a REST API.

## Installation

```bash
pip install reminix-langgraph
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from reminix_langgraph import wrap
from reminix_runtime import serve

# Create a LangGraph agent
llm = ChatOpenAI(model="gpt-4o")
graph = create_react_agent(llm, tools=[])

# Wrap it with the Reminix adapter
agent = wrap(graph, name="my-agent")

# Serve it as a REST API
serve([agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-agent/invoke` - Stateless invocation
- `POST /agents/my-agent/chat` - Conversational chat

## API Reference

### `wrap(graph, name)`

Wrap a LangGraph compiled graph for use with Reminix Runtime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `CompiledGraph` | required | A LangGraph compiled graph |
| `name` | `str` | `"langgraph-agent"` | Name for the agent (used in URL path) |

**Returns:** `LangGraphAdapter` - A Reminix adapter instance

### How It Works

LangGraph uses a state-based approach. The adapter:
1. Converts incoming messages to LangChain message format
2. Invokes the graph with `{"messages": [...]}`
3. Extracts the last AI message from the response
4. Returns it in the Reminix response format

## Runtime Documentation

For information about the server, endpoints, request/response formats, and more, see the [`reminix-runtime`](https://pypi.org/project/reminix-runtime/) package.

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## License

Apache-2.0
