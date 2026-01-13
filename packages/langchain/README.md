# reminix-langchain

Reminix adapter for LangChain agents.

## Installation

```bash
pip install reminix-langchain
```

## Usage

```python
from reminix_runtime import serve
from reminix_langchain import wrap

# Wrap your LangChain agent
wrapped_agent = wrap(agent, name="my-agent")

# Serve it
serve([wrapped_agent], port=8080)
```

## Documentation

See the [main repository](https://github.com/reminix-ai/runtime-python) for full documentation.

## License

Apache-2.0
