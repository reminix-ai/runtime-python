# Reminix Runtime (Python)

The open source runtime for serving AI agents via REST APIs. Part of [Reminix](https://reminix.com) — the developer platform for AI agents.

Deploy to [Reminix Cloud](https://reminix.com) for zero-config hosting, or self-host anywhere.

---

A lightweight runtime for serving AI agents via REST APIs. Turn any LLM framework into a REST API with built-in streaming.

**Features:**
- **REST API Server**: Execute endpoint powered by [FastAPI](https://fastapi.tiangolo.com)
- **Streaming Support**: Server-Sent Events (SSE) out of the box
- **Agent Types**: Standard patterns (prompt, chat, task, rag, thread, workflow) for common agent I/O
- **Framework Integrations**: Pre-built agents for LangChain, LangGraph, OpenAI, Anthropic, LlamaIndex

## Packages

| Package | Description |
|---------|-------------|
| [`reminix-runtime`](./packages/runtime) | Core runtime with `@agent` and `@tool` decorators and agent types |
| [`reminix-langchain`](./packages/langchain) | LangChain chat agent |
| [`reminix-langgraph`](./packages/langgraph) | LangGraph thread and workflow agents |
| [`reminix-openai`](./packages/openai) | OpenAI chat, task, and thread agents |
| [`reminix-anthropic`](./packages/anthropic) | Anthropic chat, task, and thread agents |
| [`reminix-llamaindex`](./packages/llamaindex) | LlamaIndex RAG agent |

## Installation

```bash
# Install the package for your framework (runtime is included as a dependency)
pip install reminix-langchain
```

## Quick Start

### With a Framework

```python
from langchain_openai import ChatOpenAI
from reminix_langchain import LangChainChatAgent
from reminix_runtime import serve

llm = ChatOpenAI(model="gpt-4o")
agent = LangChainChatAgent(llm, name="my-agent")
serve(agents=[agent])
```

### With Decorators (No Framework)

```python
from reminix_runtime import agent, serve

@agent
async def calculator(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

serve(agents=[calculator])
```

Your agent is now available at:
- `POST /agents/calculator/invoke` - Execute the calculator agent

See the [runtime package docs](./packages/runtime) for agent types, tools, streaming, and advanced usage.

## Using Platform Tools via MCP

When deployed to Reminix Cloud, your agents can access platform tools (memory, knowledge search) and your project tools via the MCP server. The environment variables `REMINIX_MCP_URL` and `REMINIX_API_KEY` are injected automatically.

Any framework with MCP client support works — no Reminix-specific SDK needed.

### LangChain

```python
import os
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

headers = {
    "Authorization": f"Bearer {os.environ['REMINIX_API_KEY']}",
}
# Optional: scope memory to a specific user
headers["X-Reminix-Identity"] = json.dumps({"user_id": "u_123"})

async with MultiServerMCPClient({
    "reminix": {
        "url": os.environ["REMINIX_MCP_URL"],
        "headers": headers,
    }
}) as client:
    tools = client.get_tools()
    agent = create_react_agent(ChatOpenAI(model="gpt-4o"), tools)
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "What do you know about me?"}]})
```

### OpenAI Agents SDK

```python
import os
import json
from agents import Agent
from agents.mcp import MCPServerStreamableHttp

headers = {
    "Authorization": f"Bearer {os.environ['REMINIX_API_KEY']}",
}
# Optional: scope memory to a specific user
headers["X-Reminix-Identity"] = json.dumps({"user_id": "u_123"})

mcp_server = MCPServerStreamableHttp(
    url=os.environ["REMINIX_MCP_URL"],
    headers=headers,
)

agent = Agent(
    name="my-agent",
    instructions="You are a helpful assistant.",
    mcp_servers=[mcp_server],
)
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
