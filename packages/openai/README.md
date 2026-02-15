# reminix-openai

Reminix agents for the [OpenAI API](https://platform.openai.com/docs). Serve OpenAI models as a REST API with chat, task, and thread agents.

> **Ready to go live?** [Deploy to Reminix Cloud](https://reminix.com/docs/deployment) for zero-config hosting, or [self-host](https://reminix.com/docs/deployment/self-hosting) on your own infrastructure.

## Installation

```bash
pip install reminix-openai
```

This will also install `reminix-runtime` as a dependency.

## Quick Start

### Chat Agent

The chat agent follows the chat template and supports streaming responses.

```python
from openai import AsyncOpenAI
from reminix_openai import OpenAIChatAgent
from reminix_runtime import serve

client = AsyncOpenAI()
agent = OpenAIChatAgent(client, name="my-chatbot", model="gpt-4o")
serve(agents=[agent])
```

### Task Agent

The task agent follows the task template and returns structured output. Streaming is not supported.

```python
from openai import AsyncOpenAI
from pydantic import BaseModel
from reminix_openai import OpenAITaskAgent
from reminix_runtime import serve

class Summary(BaseModel):
    title: str
    bullet_points: list[str]

client = AsyncOpenAI()
agent = OpenAITaskAgent(client, output_schema=Summary, name="summarizer", model="gpt-4o")
serve(agents=[agent])
```

### Thread Agent

The thread agent follows the thread template and supports tool use over multiple turns. Streaming is not supported.

```python
from openai import AsyncOpenAI
from reminix_openai import OpenAIThreadAgent
from reminix_runtime import serve

def get_weather(location: str) -> str:
    return f"The weather in {location} is sunny."

client = AsyncOpenAI()
agent = OpenAIThreadAgent(
    client,
    tools=[get_weather],
    name="assistant",
    model="gpt-4o",
    max_turns=10,
)
serve(agents=[agent])
```

Your agents are now available at:
- `POST /agents/{name}/invoke` - Execute the agent

## API Reference

### `OpenAIChatAgent(client, *, name, model, description, instructions)`

Create an OpenAI chat agent. Follows the chat type and supports streaming.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncOpenAI` | required | An OpenAI async client |
| `name` | `str` | `"openai-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"gpt-4o-mini"` | Model to use for completions |
| `description` | `str` | `"openai chat agent"` | Description shown in agent metadata |
| `instructions` | `str` | `None` | System instructions prepended to messages |

**Returns:** `OpenAIChatAgent` - A Reminix chat agent instance

### `OpenAITaskAgent(client, output_schema, *, name, model, description, instructions)`

Create an OpenAI task agent. Follows the task type and returns structured output. Streaming is not supported.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncOpenAI` | required | An OpenAI async client |
| `output_schema` | `type[BaseModel]` | required | A Pydantic model defining the structured output |
| `name` | `str` | `"openai-task-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"gpt-4o-mini"` | Model to use for completions |
| `description` | `str` | `"openai task agent"` | Description shown in agent metadata |
| `instructions` | `str` | `None` | System instructions prepended to messages |

**Returns:** `OpenAITaskAgent` - A Reminix task agent instance

### `OpenAIThreadAgent(client, tools, *, name, model, max_turns, description, instructions)`

Create an OpenAI thread agent. Follows the thread type and supports tool use over multiple turns. Streaming is not supported.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | `AsyncOpenAI` | required | An OpenAI async client |
| `tools` | `list` | required | A list of tool functions the agent can call |
| `name` | `str` | `"openai-thread-agent"` | Name for the agent (used in URL path) |
| `model` | `str` | `"gpt-4o-mini"` | Model to use for completions |
| `max_turns` | `int` | `10` | Maximum number of tool-use turns before stopping |
| `description` | `str` | `"openai thread agent"` | Description shown in agent metadata |
| `instructions` | `str` | `None` | System instructions prepended to messages |

**Returns:** `OpenAIThreadAgent` - A Reminix thread agent instance

### Example with Custom Configuration

```python
from openai import AsyncOpenAI
from reminix_openai import OpenAIChatAgent
from reminix_runtime import serve

client = AsyncOpenAI(
    api_key="your-api-key",
    base_url="https://your-proxy.com/v1"  # Optional: custom endpoint
)

agent = OpenAIChatAgent(
    client,
    name="gpt4-agent",
    model="gpt-4o"
)

serve(agents=[agent])
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

For streaming responses, set `stream: true` in the request (chat agent only):

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
- [OpenAI Documentation](https://platform.openai.com/docs)

## License

Apache-2.0
