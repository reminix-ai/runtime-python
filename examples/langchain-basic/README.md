# LangChain Basic Example

A simple example showing how to serve a LangChain agent via Reminix Runtime.

## Setup

```bash
# From the repository root
uv sync

# Navigate to this example
cd examples/langchain-basic
```

## Usage

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your-api-key

# Run the example
uv run python main.py
```

## How it works

1. Create a LangChain agent using `langchain-openai`
2. Wrap it with `reminix-langchain`
3. Serve it with `reminix-runtime`

```python
from reminix_runtime import serve
from reminix_langchain import wrap

agent = create_react_agent(model, tools=[...])
wrapped_agent = wrap(agent, name="my-agent")

serve([wrapped_agent], port=8080)
```
