# reminix-llamaindex

Reminix adapter for LlamaIndex agents.

## Installation

```bash
pip install reminix-llamaindex
```

## Usage

```python
from reminix_runtime import serve
from reminix_llamaindex import wrap

# Wrap your LlamaIndex agent
wrapped_agent = wrap(agent, name="my-agent")

# Serve it
serve([wrapped_agent], port=8080)
```

## Documentation

See the [main repository](https://github.com/reminix-ai/runtime-python) for full documentation.

## License

Apache-2.0
