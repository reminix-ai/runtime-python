# Anthropic SDK Agent

Agent using the Anthropic SDK directly. Claude models with tool use and streaming — no framework overhead.

[![Deploy to Reminix](https://reminix.com/badge/deploy.svg)](https://reminix.com/new/deploy?repo=reminix-ai/runtime-python&folder=examples/anthropic-agent)

## Required environment variables

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your [Anthropic API key](https://console.anthropic.com/) |

## How it works

`AnthropicChatAgent` wraps an Anthropic SDK client so Reminix can invoke it through the uniform `Agent` interface. The adapter translates Reminix invoke requests into Anthropic Messages API calls and streams responses back — you get tool use, system prompts, and message history handling without writing any protocol glue.

## What it does

```python
from anthropic import AsyncAnthropic

from reminix_anthropic import AnthropicChatAgent
from reminix_runtime import serve

agent = AnthropicChatAgent(
    AsyncAnthropic(),
    name="anthropic-agent",
    model="claude-sonnet-4-6",
)

if __name__ == "__main__":
    serve(agents=[agent])
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/manifest` | GET | Agent discovery |
| `/agents/anthropic-agent/invoke` | POST | Execute the agent |

## Testing

```bash
# Single prompt
curl -X POST http://localhost:8080/agents/anthropic-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is the capital of France?"}}'

# Chat-style messages
curl -X POST http://localhost:8080/agents/anthropic-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'
```

## Run locally

```bash
uv sync
ANTHROPIC_API_KEY=... uv run python main.py
```
