# Reminix Runtime (Python)

The open source runtime for serving AI agents via REST APIs. Part of [Reminix](https://reminix.com) — the developer platform for AI agents.

Deploy to [Reminix Cloud](https://reminix.com) for zero-config hosting, or self-host anywhere.

---

A lightweight runtime for serving AI agents via REST APIs. Wrap any LLM framework and get an execute endpoint with built-in streaming.

**Features:**
- **REST API Server**: Execute endpoint powered by [FastAPI](https://fastapi.tiangolo.com)
- **Streaming Support**: Server-Sent Events (SSE) out of the box
- **Framework Adapters**: Pre-built integrations for LangChain, LangGraph, OpenAI, Anthropic, LlamaIndex

## Packages

| Package | Description |
|---------|-------------|
| [`reminix-runtime`](./packages/runtime) | Core runtime with `@agent`, `@chat_agent`, and `@tool` decorators |
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
from reminix_langchain import wrap_agent
from reminix_runtime import serve

agent = ChatOpenAI(model="gpt-4o")

serve(agents=[wrap_agent(agent, name="my-agent")], port=8080)
```

### With Decorators (No Framework)

```python
from reminix_runtime import agent, chat_agent, serve, Message

@agent
async def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

@chat_agent
async def assistant(messages: list[Message]) -> str:
    """A helpful assistant."""
    return f"You said: {messages[-1].content}"

serve(agents=[calculator, assistant], port=8080)
```

Your agents are now available at:
- `POST /agents/calculator/invoke` - Execute the calculator agent
- `POST /agents/assistant/invoke` - Execute the assistant agent

See the [runtime package docs](./packages/runtime) for tools, streaming, and advanced usage.

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

### Code Quality

```bash
# Format code (auto-fix)
uv run ruff format .
uv run ruff check --fix .

# Check formatting (CI)
uv run ruff format --check .

# Lint code
uv run ruff check .

# Type check
uv run pyright

# Run all checks (before pushing)
uv run check

# Run all checks + tests (before pushing)
uv run prepush
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

Apache-2.0
