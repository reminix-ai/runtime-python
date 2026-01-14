# reminix-llamaindex

Reminix Runtime adapter for [LlamaIndex](https://www.llamaindex.ai/). Deploy LlamaIndex chat engines as a REST API.

## Installation

```bash
pip install reminix-llamaindex
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.llms.openai import OpenAI
from reminix_llamaindex import wrap
from reminix_runtime import serve

# Create a LlamaIndex chat engine
llm = OpenAI(model="gpt-4o")
engine = SimpleChatEngine.from_defaults(llm=llm)

# Wrap it with the Reminix adapter
agent = wrap(engine, name="my-chatbot")

# Serve it as a REST API
serve([agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-chatbot/invoke` - Stateless invocation
- `POST /agents/my-chatbot/chat` - Conversational chat

## API Reference

### `wrap(engine, name)`

Wrap a LlamaIndex chat engine for use with Reminix Runtime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `BaseChatEngine` | required | A LlamaIndex chat engine |
| `name` | `str` | `"llamaindex-agent"` | Name for the agent (used in URL path) |

**Returns:** `LlamaIndexAdapter` - A Reminix adapter instance

### Example with RAG

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.openai import OpenAI
from reminix_llamaindex import wrap
from reminix_runtime import serve

# Load documents and create index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Create a chat engine with the index
engine = index.as_chat_engine(llm=OpenAI(model="gpt-4o"))

# Wrap and serve
agent = wrap(engine, name="rag-chatbot")
serve([agent], port=8080)
```

## Endpoint Input/Output Formats

### POST /agents/{name}/invoke

Stateless invocation for task-oriented operations.

**Request:**
```json
{
  "input": {
    "query": "What is the capital of France?"
  }
}
```

Or with prompt:
```json
{
  "input": {
    "prompt": "Summarize this text: ..."
  }
}
```

**Response:**
```json
{
  "output": "The capital of France is Paris."
}
```

### POST /agents/{name}/chat

Conversational chat. The adapter extracts the last user message.

**Request:**
```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

**Response:**
```json
{
  "output": "The capital of France is Paris.",
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

## Runtime Documentation

For information about the server, endpoints, request/response formats, and more, see the [`reminix-runtime`](https://pypi.org/project/reminix-runtime/) package.

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [LlamaIndex Documentation](https://docs.llamaindex.ai/)

## License

Apache-2.0
