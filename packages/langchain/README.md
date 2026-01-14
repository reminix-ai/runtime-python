# reminix-langchain

Reminix Runtime adapter for [LangChain](https://langchain.com). Deploy any LangChain runnable as a REST API.

## Installation

```bash
pip install reminix-langchain
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from langchain_openai import ChatOpenAI
from reminix_langchain import wrap
from reminix_runtime import serve

# Create a LangChain model or chain
llm = ChatOpenAI(model="gpt-4o")

# Wrap it with the Reminix adapter
agent = wrap(llm, name="my-chatbot")

# Serve it as a REST API
serve([agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-chatbot/invoke` - Stateless invocation
- `POST /agents/my-chatbot/chat` - Conversational chat

## API Reference

### `wrap(runnable, name)`

Wrap a LangChain runnable for use with Reminix Runtime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `runnable` | `Runnable` | required | Any LangChain runnable (LLM, chain, agent, etc.) |
| `name` | `str` | `"langchain-agent"` | Name for the agent (used in URL path) |

**Returns:** `LangChainAdapter` - A Reminix adapter instance

### Example with a Chain

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from reminix_langchain import wrap
from reminix_runtime import serve

# Create a chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm

# Wrap and serve
agent = wrap(chain, name="my-chain")
serve([agent], port=8080)
```

## Endpoint Input/Output Formats

### POST /agents/{name}/invoke

Stateless invocation for task-oriented operations.

**Request:**
```json
{
  "input": {
    "prompt": "Summarize this text: ..."
  }
}
```

Or with messages:
```json
{
  "input": {
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }
}
```

**Response:**
```json
{
  "output": "Hello! How can I help you today?"
}
```

### POST /agents/{name}/chat

Conversational chat with message history.

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
- [LangChain Documentation](https://python.langchain.com)

## License

Apache-2.0
