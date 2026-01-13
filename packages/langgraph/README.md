# reminix-langgraph

Reminix adapter for LangGraph agents.

## Installation

```bash
pip install reminix-langgraph
```

## Usage

```python
from reminix_runtime import serve
from reminix_langgraph import wrap

# Wrap your LangGraph agent
wrapped_agent = wrap(graph, name="my-agent")

# Serve it
serve([wrapped_agent], port=8080)
```

## Documentation

See the [main repository](https://github.com/reminix-ai/runtime-python) for full documentation.

## License

Apache-2.0
