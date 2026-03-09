"""
Runtime Tools Example

This example shows how to create and serve standalone tools using
the @tool decorator from reminix_runtime.

Usage:
    uv run python main.py

Then test the endpoints:

    # Health check
    curl http://localhost:8080/health

    # Discovery
    curl http://localhost:8080/manifest

    # List tools via MCP
    curl -X POST http://localhost:8080/mcp \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc": "2.0", "method": "tools/list", "id": 1}'

    # Call a tool via MCP
    curl -X POST http://localhost:8080/mcp \
      -H "Content-Type: application/json" \
      -d '{"jsonrpc": "2.0", "method": "tools/call", "params": {"name": "get_weather", "arguments": {"location": "San Francisco"}}, "id": 2}'
"""

from reminix_runtime import serve, tool

# Weather data (simulated)
WEATHER_DATA: dict[str, dict[str, str | int]] = {
    "san francisco": {"temp": 65, "condition": "foggy"},
    "new york": {"temp": 45, "condition": "cloudy"},
    "los angeles": {"temp": 75, "condition": "sunny"},
    "seattle": {"temp": 50, "condition": "rainy"},
    "miami": {"temp": 85, "condition": "humid"},
}


@tool
async def get_weather(location: str, units: str = "fahrenheit") -> dict:
    """Get the current weather for a city.

    Args:
        location: City name (e.g., "San Francisco", "New York")
        units: Temperature units - "celsius" or "fahrenheit"
    """
    location_lower = location.lower()

    weather = WEATHER_DATA.get(location_lower)
    if not weather:
        return {
            "error": f'Weather data not available for "{location}"',
            "available_cities": list(WEATHER_DATA.keys()),
        }

    temp = weather["temp"]
    if units == "celsius":
        temp = round((int(temp) - 32) * 5 / 9)

    return {
        "location": location,
        "temperature": temp,
        "units": units,
        "condition": weather["condition"],
    }


@tool
def calculate(a: float, b: float, operation: str) -> dict:
    """Perform basic math operations.

    Args:
        a: First operand
        b: Second operand
        operation: Math operation - "add", "subtract", "multiply", or "divide"
    """
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return {"error": "Cannot divide by zero"}
        result = a / b
    else:
        return {"error": f"Unknown operation: {operation}"}

    return {"a": a, "b": b, "operation": operation, "result": result}


@tool
def string_utils(text: str, operation: str) -> dict:
    """Perform string operations.

    Args:
        text: Input text
        operation: String operation - "uppercase", "lowercase", "reverse", or "length"
    """
    if operation == "uppercase":
        return {"result": text.upper()}
    elif operation == "lowercase":
        return {"result": text.lower()}
    elif operation == "reverse":
        return {"result": text[::-1]}
    elif operation == "length":
        return {"result": len(text)}
    else:
        return {"error": f"Unknown operation: {operation}"}


if __name__ == "__main__":
    print("Runtime Tools Example")
    print("=" * 40)
    print("Tools:")
    print("  - get_weather: Get weather for a city")
    print("  - calculate: Basic math operations")
    print("  - string_utils: String manipulation")
    print()
    print("Server running on http://localhost:8080")
    print()
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /manifest")
    print("  POST /mcp (MCP Streamable HTTP - tool discovery and execution)")
    print()

    serve(tools=[get_weather, calculate, string_utils])
