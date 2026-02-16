# reminix-anthropic

Reminix agents for the [Anthropic API](https://docs.anthropic.com/). Serve Claude models as a REST API.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

## Installation

```bash
pip install reminix-anthropic
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

### Chat Agent (streaming conversations)

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import AnthropicChatAgent
from reminix_runtime import serve

client = AsyncAnthropic()
agent = AnthropicChatAgent(client, name="my-claude")
serve(agents=[agent])
```

### Task Agent (structured output)

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import AnthropicTaskAgent
from reminix_runtime import serve

client = AsyncAnthropic()
schema = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
        "confidence": {"type": "number"},
    },
    "required": ["sentiment", "confidence"],
}
agent = AnthropicTaskAgent(client, output_schema=schema, name="sentiment-analyzer")
serve(agents=[agent])
```

### Thread Agent (tool-calling loop)

```python
from anthropic import AsyncAnthropic
from reminix_anthropic import AnthropicThreadAgent
from reminix_runtime import serve, tool

@tool(name="get_weather", description="Get the current weather for a city")
async def get_weather(city: str) -> dict:
    return {"temperature": 72, "condition": "sunny"}

client = AsyncAnthropic()
agent = AnthropicThreadAgent(client, tools=[get_weather], name="weather-assistant")
serve(agents=[agent])
```

Your agents are now available at:
- `POST /agents/{name}/invoke` - Execute the agent

## API Reference

### `AnthropicChatAgent(client, *, name, model, max_tokens, description, instructions, tags, metadata)`

Create an Anthropic chat agent. Supports streaming.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncAnthropic` | required | An Anthropic async client |
| `name` | `str` | `"anthropic-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model to use |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `description` | `str` | `"anthropic chat agent"` | Description shown in agent metadata |
| `instructions` | `str` | `None` | System instructions merged with system messages |
| `tags` | `list[str]` | `None` | Tags for categorizing/filtering agents |
| `metadata` | `dict` | `None` | Custom metadata merged into agent info |

**Returns:** `AnthropicChatAgent` - A Reminix chat agent instance

The chat agent:
1. Converts incoming messages to Anthropic format
2. Extracts system messages and merges with `instructions` as the `system` parameter
3. Returns the assistant's text response
4. Supports streaming via Server-Sent Events

### `AnthropicTaskAgent(client, output_schema, *, name, model, max_tokens, description, instructions, tags, metadata)`

Create an Anthropic task agent. Returns structured output via tool-use. Does not support streaming.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncAnthropic` | required | An Anthropic async client |
| `output_schema` | `dict` | required | JSON Schema defining the structured output |
| `name` | `str` | `"anthropic-task-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model to use |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `description` | `str` | `"anthropic task agent"` | Description shown in agent metadata |
| `instructions` | `str` | `None` | System instructions passed as `system` parameter |
| `tags` | `list[str]` | `None` | Tags for categorizing/filtering agents |
| `metadata` | `dict` | `None` | Custom metadata merged into agent info |

**Returns:** `AnthropicTaskAgent` - A Reminix task agent instance

The task agent:
1. Reads the `task` field from the request input
2. Includes any additional input fields as context
3. Forces a tool call using the provided `output_schema`
4. Extracts and returns the structured result from the tool-use block

### `AnthropicThreadAgent(client, tools, *, name, model, max_tokens, max_turns, description, instructions, tags, metadata)`

Create an Anthropic thread agent with a tool-calling loop. Does not support streaming.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncAnthropic` | required | An Anthropic async client |
| `tools` | `list[Tool]` | required | List of tools available to the agent |
| `name` | `str` | `"anthropic-thread-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"claude-sonnet-4-20250514"` | Model to use |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `max_turns` | `int` | `10` | Maximum number of tool-calling turns |
| `description` | `str` | `"anthropic thread agent"` | Description shown in agent metadata |
| `instructions` | `str` | `None` | System instructions merged with system messages |
| `tags` | `list[str]` | `None` | Tags for categorizing/filtering agents |
| `metadata` | `dict` | `None` | Custom metadata merged into agent info |

**Returns:** `AnthropicThreadAgent` - A Reminix thread agent instance

The thread agent:
1. Converts incoming messages to Anthropic format
2. Calls the model in a loop, executing tool calls each turn
3. Continues until the model produces a final response or `max_turns` is reached
4. Returns the full conversation including tool calls and results

### System Messages

All three agents automatically handle Anthropic's system message format. System messages in your request are extracted and passed as the `system` parameter to the API.

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

### Chat Agent -- POST /agents/{name}/invoke

**Request with prompt:**
```json
{
  "input": {
    "prompt": "Summarize this text: ..."
  }
}
```

**Request with messages:**
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

### Task Agent -- POST /agents/{name}/invoke

**Request:**
```json
{
  "input": {
    "task": "Analyze the sentiment of this review: 'Great product, love it!'"
  }
}
```

**Response:**
```json
{
  "output": {
    "sentiment": "positive",
    "confidence": 0.95
  }
}
```

### Thread Agent -- POST /agents/{name}/invoke

**Request:**
```json
{
  "input": {
    "messages": [
      {"role": "user", "content": "What's the weather in San Francisco?"}
    ]
  }
}
```

**Response:**
```json
{
  "output": [
    {"role": "user", "content": "What's the weather in San Francisco?"},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "toolu_01...", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"San Francisco\"}"}}]},
    {"role": "tool", "content": "{\"temperature\": 72, \"condition\": \"sunny\"}", "tool_call_id": "toolu_01..."},
    {"role": "assistant", "content": "The weather in San Francisco is 72 degrees and sunny."}
  ]
}
```

### Streaming

For streaming responses (chat agent only), set `stream: true` in the request:

```json
{
  "input": {
    "prompt": "Tell me a story"
  },
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
