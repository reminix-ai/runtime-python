# Reminix Runtime (Python)

Your AI agent, live in minutes. Wrap your LangChain, OpenAI, Anthropic, or LlamaIndex agent in one line — deploy and start streaming responses to your users.

**Features:**
- **Instant APIs**: Invoke and chat endpoints out of the box
- **Real-Time Streaming**: Built-in SSE support
- **Bring Your Framework**: Adapters for LangChain, LangGraph, OpenAI, Anthropic, LlamaIndex

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
