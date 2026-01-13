# reminix-anthropic

Reminix adapter for Anthropic agents.

## Installation

```bash
pip install reminix-anthropic
```

## Usage

```python
from reminix_runtime import serve
from reminix_anthropic import wrap

# Wrap your Anthropic agent
wrapped_agent = wrap(agent, name="my-agent")

# Serve it
serve([wrapped_agent], port=8080)
```

## Documentation

See the [main repository](https://github.com/reminix-ai/runtime-python) for full documentation.

## License

Apache-2.0
