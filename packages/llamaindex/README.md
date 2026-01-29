# reminix-llamaindex

Reminix Runtime adapter for [LlamaIndex](https://www.llamaindex.ai/). Serve LlamaIndex chat engines as a REST API.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

## Installation

```bash
pip install reminix-llamaindex
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from reminix_llamaindex import serve_agent

llm = OpenAI(model="gpt-4o")
engine = SimpleChatEngine.from_defaults(llm=llm)
serve_agent(engine, name="my-chatbot", port=8080)
```

For more flexibility (e.g., serving multiple agents), use `wrap_agent` and `serve` separately:

```python
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from reminix_llamaindex import wrap_agent
from reminix_runtime import serve

llm = OpenAI(model="gpt-4o")
engine = SimpleChatEngine.from_defaults(llm=llm)
agent = wrap_agent(engine, name="my-chatbot")
serve(agents=[agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-chatbot/invoke` - Execute the agent

## API Reference

### `serve_agent(engine, name, port, host)`

Wrap a LlamaIndex chat engine and serve it immediately. Combines `wrap_agent` and `serve` for single-agent setups.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `BaseChatEngine` | required | A LlamaIndex chat engine |
| `name` | `str` | `"llamaindex-agent"` | Name for the agent (used in URL path) |
| `port` | `int` | `8080` | Port to serve on |
| `host` | `str` | `"0.0.0.0"` | Host to bind to |

### `wrap_agent(engine, name)`

Wrap a LlamaIndex chat engine for use with Reminix Runtime. Use this with `serve` from `reminix_runtime` for multi-agent setups.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `BaseChatEngine` | required | A LlamaIndex chat engine |
| `name` | `str` | `"llamaindex-agent"` | Name for the agent (used in URL path) |

**Returns:** `LlamaIndexAgentAdapter` - A Reminix adapter instance

### Example with RAG

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from reminix_llamaindex import wrap_agent
from reminix_runtime import serve

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create a chat engine with the index
engine = index.as_chat_engine(llm=OpenAI(model="gpt-4o"))

# Wrap and serve
agent = wrap_agent(engine, name="rag-chatbot")
serve(agents=[agent], port=8080)
```

## Endpoint Input/Output Formats

### POST /agents/{name}/invoke

Execute the agent with a query or prompt.

**Request with query:**
```json
{
  "query": "What is the capital of France?"
}
```

**Request with prompt:**
```json
{
  "prompt": "Summarize this text: ..."
}
```

**Response:**
```json
{
  "output": "The capital of France is Paris."
}
```

### Streaming

For streaming responses, set `stream: true` in the request:

```json
{
  "query": "Tell me about Paris",
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
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

## License

Apache-2.0
