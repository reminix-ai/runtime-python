# Runtime Tools Example

An example showing how to serve standalone tools via Reminix Runtime without any AI framework.

## Setup

```bash
# From the repository root
uv sync

# Navigate to this example
cd examples/runtime-tools
```

## Usage

```bash
uv run python main.py
```

## Endpoints

Once running, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/info` | GET | Tool discovery |
| `/tools/get_weather/execute` | POST | Get weather for a city |
| `/tools/calculate/execute` | POST | Basic math operations |
| `/tools/string_utils/execute` | POST | String manipulation |

## Available Tools

- `get_weather(location, units?)`: Get weather for San Francisco, New York, Los Angeles, Seattle, or Miami
- `calculate(a, b, operation)`: Perform add, subtract, multiply, or divide
- `string_utils(text, operation)`: Perform uppercase, lowercase, reverse, or length

## Testing

```bash
# Health check
curl http://localhost:8080/health

# Discovery (shows all tools with their schemas)
curl http://localhost:8080/info

# Get weather
curl -X POST http://localhost:8080/tools/get_weather/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"location": "San Francisco"}}'

# Get weather in Celsius
curl -X POST http://localhost:8080/tools/get_weather/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"location": "New York", "units": "celsius"}}'

# Calculate
curl -X POST http://localhost:8080/tools/calculate/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"a": 10, "b": 5, "operation": "multiply"}}'

# String utils
curl -X POST http://localhost:8080/tools/string_utils/execute \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "Hello World", "operation": "reverse"}}'
```

## How it works

1. Define tools using `@tool` decorator from `reminix_runtime`
2. The decorator extracts name, description, and parameters from the function
3. Serve tools with `serve(tools=[...])`

```python
from reminix_runtime import tool, serve

@tool
async def get_weather(location: str, units: str = "fahrenheit") -> dict:
    """Get the current weather for a city."""
    # Fetch weather data...
    return {"location": location, "temperature": 72, "condition": "sunny"}

serve(tools=[get_weather], port=8080)
```
