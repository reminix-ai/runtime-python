# Reminix Runtime (Python)

Deploy AI agents as REST APIs. Lightweight runtime with adapters for LangChain, LangGraph, OpenAI, Anthropic, and more.

**Two interaction modes:**
- **Invoke** (Stateless) - Single request/response
- **Chat** (Conversational) - With message history

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

```python
from langchain.agents import create_agent
from reminix_langchain import wrap
from reminix_runtime import serve

# Create your agent with any framework
agent = create_agent(model="gpt-4o", tools=[...])

# Serve it via Reminix
serve([wrap(agent)], port=8080)
```

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
# Run all tests across all packages
uv run --extra dev pytest

# Run tests for a specific package
cd packages/runtime
uv run --extra dev pytest
```

### Building

```bash
uv build
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

Apache-2.0
