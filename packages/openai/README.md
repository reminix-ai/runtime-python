# reminix-openai

Reminix adapter for OpenAI Agents.

## Installation

```bash
pip install reminix-openai
```

## Usage

```python
from reminix_runtime import serve
from reminix_openai import wrap

# Wrap your OpenAI agent
wrapped_agent = wrap(agent, name="my-agent")

# Serve it
serve([wrapped_agent], port=8080)
```

## Documentation

See the [main repository](https://github.com/reminix-ai/runtime-python) for full documentation.

## License

Apache-2.0
