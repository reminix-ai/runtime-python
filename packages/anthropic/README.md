# reminix-anthropic

Reminix Runtime adapter for the [Anthropic API](https://docs.anthropic.com/). Deploy Claude models as a REST API.

## Installation

```bash
pip install reminix-anthropic
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import wrap
from reminix_runtime import serve

# Create an Anthropic client
client = AsyncAnthropic()

# Wrap it with the Reminix adapter
agent = wrap(client, name="my-claude", model="claude-sonnet-4-20250514")

# Serve it as a REST API
serve([agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-claude/invoke` - Stateless invocation
- `POST /agents/my-claude/chat` - Conversational chat

## API Reference

### `wrap(client, name, model, max_tokens)`

Wrap an Anthropic client for use with Reminix Runtime.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncAnthropic` | required | An Anthropic async client |
| `name` | `str` | `"anthropic-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model to use |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |

**Returns:** `AnthropicAdapter` - A Reminix adapter instance

### System Messages

The adapter automatically handles Anthropic's system message format. System messages in your request are extracted and passed as the `system` parameter to the API.

```python
# This works automatically:
request = {
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello!"}
    ]
}
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
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ]
}
```

**Response:**
```json
{
  "output": "The capital of France is Paris.",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

## Runtime Documentation

For information about the server, endpoints, request/response formats, and more, see the [`reminix-runtime`](https://pypi.org/project/reminix-runtime/) package.

## Links

- [GitHub Repository](https://github.com/reminix-ai/runtime-python)
- [Anthropic Documentation](https://docs.anthropic.com/)

## License

Apache-2.0
