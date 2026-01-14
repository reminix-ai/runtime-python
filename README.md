# Reminix Runtime (Python)

A lightweight runtime for serving AI agents via REST APIs. Wrap any LLM framework and get invoke/chat endpoints with built-in streaming.

**Features:**
- **REST API Server**: Invoke and chat endpoints powered by [FastAPI](https://fastapi.tiangolo.com)
- **Streaming Support**: Server-Sent Events (SSE) out of the box
- **Framework Adapters**: Pre-built integrations for LangChain, LangGraph, OpenAI, Anthropic, LlamaIndex

## Packages

| Package | Description |
|---------|-------------|
| [`reminix-runtime`](./packages/runtime) | Core runtime with `serve()`, invoke/chat handlers, and base adapter |
| [`reminix-langchain`](./packages/langchain) | LangChain adapter |
| [`reminix-langgraph`](./packages/langgraph) | LangGraph adapter |
| [`reminix-openai`](./packages/openai) | OpenAI Agents adapter |
| [`reminix-anthropic`](./packages/anthropic) | Anthropic adapter |
| [`reminix-llamaindex`](./packages/llamaindex) | LlamaIndex adapter |

## Installation

```bash
# Install the adapter for your framework (runtime is included as a dependency)
pip install reminix-langchain
```

## Quick Start

### With a Framework

```python
from langchain_openai import ChatOpenAI
from reminix_langchain import wrap
from reminix_runtime import serve

agent = ChatOpenAI(model="gpt-4o")

serve([wrap(agent, name="my-agent")], port=8080)
```

### With Decorators (No Framework)

```python
from reminix_runtime import Agent, serve

agent = Agent("my-agent")

@agent.on_invoke
async def handle_invoke(request):
    return {"output": f"Received: {request.input}"}

@agent.on_chat
async def handle_chat(request):
    last_message = request.messages[-1].content if request.messages else ""
    return {
        "output": f"You said: {last_message}",
        "messages": [*request.messages, {"role": "assistant", "content": f"You said: {last_message}"}]
    }

serve([agent], port=8080)
```

Your agent is now available at:
- `POST /agents/my-agent/invoke` - Stateless invocation
- `POST /agents/my-agent/chat` - Conversational chat

## Development

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Setup

```bash
# Clone the repository
git clone https://github.com/reminix-ai/runtime-python.git
cd runtime-python

# Install dependencies
uv sync

# Or with pip
pip install -e packages/runtime
pip install -e packages/langchain
# ... etc
```

### Running Examples

```bash
# Run the LangChain example
cd examples/langchain-basic
uv run python main.py
```

See the [examples/](./examples) directory for more.

### Running Tests

```bash
# Install all packages with dev dependencies
uv sync --all-packages --extra dev

# Run all tests across all packages
uv run --extra dev pytest

# Run tests for a specific package
cd packages/runtime
uv run --extra dev pytest
```

### Running Integration Tests

Integration tests require API keys. Create a `.env` file from the example:

```bash
cp .env.example .env
# Edit .env with your API keys
```

Then run:

```bash
# Run all integration tests
uv run --extra dev pytest tests/integration -v

# Run OpenAI integration tests only
uv run --extra dev pytest tests/integration/test_openai.py -v

# Run Anthropic integration tests only
uv run --extra dev pytest tests/integration/test_anthropic.py -v
```

### Building

```bash
uv build
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

Apache-2.0
