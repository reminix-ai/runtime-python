# Google Gemini Agent

Agent using Google Gemini. Flash, Pro, and Ultra models with tool use and streaming — no framework overhead.

[![Deploy to Reminix](https://reminix.com/badge/deploy.svg)](https://reminix.com/new/deploy?repo=reminix-ai/runtime-python&folder=examples/gemini-agent)

## Required environment variables

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Your [Google AI API key](https://aistudio.google.com/apikey) |

## How it works

`GoogleChatAgent` wraps a `genai.Client` so Reminix can invoke it through the uniform `Agent` interface. The adapter translates Reminix invoke requests into Gemini generate-content calls and streams responses back — you get tool use, system prompts, and message history handling without writing any protocol glue.

## What it does

```python
from google import genai

from reminix_google import GoogleChatAgent
from reminix_runtime import serve

agent = GoogleChatAgent(
    genai.Client(),
    name="gemini-agent",
    model="gemini-2.5-flash",
)

if __name__ == "__main__":
    serve(agents=[agent])
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/manifest` | GET | Agent discovery |
| `/agents/gemini-agent/invoke` | POST | Execute the agent |

## Testing

```bash
# Single prompt
curl -X POST http://localhost:8080/agents/gemini-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is the capital of France?"}}'

# Chat-style messages
curl -X POST http://localhost:8080/agents/gemini-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'
```

## Run locally

```bash
uv sync
GOOGLE_API_KEY=... uv run python main.py
```
