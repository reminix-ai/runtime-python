# reminix-openai

Reminix Runtime adapter for the [OpenAI API](https://platform.openai.com/docs). Deploy OpenAI models as a REST API.

## Installation

```bash
pip install reminix-openai
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from openai import AsyncOpenAI
from reminix_openai import wrap
from reminix_runtime import serve

# Create an OpenAI client
client = AsyncOpenAI()

# Wrap it with the Reminix adapter
agent = wrap(client, name="my-chatbot", model="gpt-4o")

# Serve it as a REST API
serve([agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-chatbot/invoke` - Stateless invocation
- `POST /agents/my-chatbot/chat` - Conversational chat

## API Reference

### `wrap(client, name, model, max_tokens)`

Wrap an OpenAI client for use with Reminix Runtime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncOpenAI` | required | An OpenAI async client |
| `name` | `str` | `"openai-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"gpt-4o-mini"` | Model to use for completions |

**Returns:** `OpenAIAdapter` - A Reminix adapter instance

### Example with Custom Configuration

```python
from openai import AsyncOpenAI
from reminix_openai import wrap
from reminix_runtime import serve

client = AsyncOpenAI(
    api_key="your-api-key",
    base_url="https://your-proxy.com/v1"  # Optional: custom endpoint
)

agent = wrap(
    client,
    name="gpt4-agent",
    model="gpt-4o"
)

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
      {"role": "system", "content": "You are a helpful assistant."},
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
- [OpenAI Documentation](https://platform.openai.com/docs)

## License

Apache-2.0
