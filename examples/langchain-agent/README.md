# LangChain Agent

Agent built with LangChain. Chains, retrievers, and tool use — served as a production REST API.

[![Deploy to Reminix](https://reminix.com/badge/deploy.svg)](https://reminix.com/new/deploy?repo=reminix-ai/runtime-python&folder=examples/langchain-agent)

## Required environment variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your [OpenAI API key](https://platform.openai.com/api-keys) |

## How it works

`LangChainChatAgent` adapts any LangChain Runnable — a chat model, a chain, or an agent — into a Reminix-compatible agent. Swap the `ChatOpenAI` below for your own composition (retrieval chain, multi-step agent, RAG pipeline) and it deploys the same way. Reminix handles streaming, message-history mapping, and the API surface.

## What it does

```python
from langchain_openai import ChatOpenAI

from reminix_langchain import LangChainChatAgent
from reminix_runtime import serve

agent = LangChainChatAgent(
    ChatOpenAI(model="gpt-4o-mini"),
    name="langchain-agent",
)

if __name__ == "__main__":
    serve(agents=[agent])
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/manifest` | GET | Agent discovery |
| `/agents/langchain-agent/invoke` | POST | Execute the agent |

## Testing

```bash
# Single prompt
curl -X POST http://localhost:8080/agents/langchain-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is AI?"}}'

# Chat-style messages
curl -X POST http://localhost:8080/agents/langchain-agent/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'
```

## Run locally

```bash
uv sync
OPENAI_API_KEY=... uv run python main.py
```
