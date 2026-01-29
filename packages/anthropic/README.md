# reminix-anthropic

Reminix Runtime adapter for the [Anthropic API](https://docs.anthropic.com/). Serve Claude models as a REST API.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

## Installation

```bash
pip install reminix-anthropic
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import serve_agent

client = AsyncAnthropic()
serve_agent(client, name="my-claude", model="claude-sonnet-4-20250514", port=8080)
```

For more flexibility (e.g., serving multiple agents), use `wrap_agent` and `serve` separately:

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import wrap_agent
from reminix_runtime import serve

client = AsyncAnthropic()
agent = wrap_agent(client, name="my-claude", model="claude-sonnet-4-20250514")
serve(agents=[agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-claude/invoke` - Execute the agent

## API Reference

### `serve_agent(client, name, model, max_tokens, port, host)`

Wrap an Anthropic client and serve it immediately. Combines `wrap_agent` and `serve` for single-agent setups.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncAnthropic` | required | An Anthropic async client |
| `name` | `str` | `"anthropic-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model to use |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `port` | `int` | `8080` | Port to serve on |
| `host` | `str` | `"0.0.0.0"` | Host to bind to |

### `wrap_agent(client, name, model, max_tokens)`

Wrap an Anthropic client for use with Reminix Runtime. Use this with `serve` from `reminix_runtime` for multi-agent setups.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncAnthropic` | required | An Anthropic async client |
| `name` | `str` | `"anthropic-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model to use |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |

**Returns:** `AnthropicAgentAdapter` - A Reminix adapter instance

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

Execute the agent with a prompt or messages.

**Request with prompt:**
```json
{
  "prompt": "Summarize this text: ..."
}
```

**Request with messages:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
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
  "prompt": "Tell me a story",
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
- [Anthropic Documentation](https://docs.anthropic.com/)

## License

Apache-2.0
