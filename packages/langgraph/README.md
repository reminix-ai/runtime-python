# reminix-langgraph

Reminix Runtime adapter for [LangGraph](https://langchain-ai.github.io/langgraph/). Serve any LangGraph agent as a REST API.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

## Installation

```bash
pip install reminix-langgraph
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from reminix_langgraph import serve_agent

llm = ChatOpenAI(model="gpt-4o")
graph = create_react_agent(llm, tools=[])
serve_agent(graph, name="my-agent", port=8080)
```

For more flexibility (e.g., serving multiple agents), use `wrap_agent` and `serve` separately:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from reminix_langgraph import wrap_agent
from reminix_runtime import serve

llm = ChatOpenAI(model="gpt-4o")
graph = create_react_agent(llm, tools=[])
agent = wrap_agent(graph, name="my-agent")
serve(agents=[agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-agent/invoke` - Execute the agent

## API Reference

### `serve_agent(graph, name, port, host)`

Wrap a LangGraph graph and serve it immediately. Combines `wrap_agent` and `serve` for single-agent setups.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `CompiledGraph` | required | A LangGraph compiled graph |
| `name` | `str` | `"langgraph-agent"` | Name for the agent (used in URL path) |
| `port` | `int` | `8080` | Port to serve on |
| `host` | `str` | `"0.0.0.0"` | Host to bind to |

### `wrap_agent(graph, name)`

Wrap a LangGraph compiled graph for use with Reminix Runtime. Use this with `serve` from `reminix_runtime` for multi-agent setups.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph` | `CompiledGraph` | required | A LangGraph compiled graph |
| `name` | `str` | `"langgraph-agent"` | Name for the agent (used in URL path) |

**Returns:** `LangGraphAgentAdapter` - A Reminix adapter instance

### How It Works

LangGraph uses a state-based approach. The adapter:
1. Converts incoming messages to LangChain message format
2. Invokes the graph with `{"messages": [...]}`
3. Extracts the last AI message from the response
4. Returns it in the Reminix response format

## Endpoint Input/Output Formats

### POST /agents/{name}/invoke

Execute the graph. Input keys are passed directly to the graph.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"}
  ]
}
```

**Response:**
```json
{
  "output": "Hello! How can I help you today?"
}
```

### Streaming

For streaming responses, set `stream: true` in the request:

```json
{
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true
}
```

The response will be sent as Server-Sent Events (SSE).

## Runtime Documentation

For information about the server, endpoints, request/response formats, and more, see the [`reminix-runtime`](https://pypi.org/project/reminix-runtime/) package.

## Deployment

Ready to go live?

- **[Deploy to Reminix Cloud](https://reminix.com/docs/deployment)** - Zero-config cloud hosting
- **[Self-host](https://reminix.com/docs/deployment/self-hosting)** - Run on your own infrastructure

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## License

Apache-2.0
