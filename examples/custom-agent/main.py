"""
Custom Agent Example

This example shows how to create a custom agent using the @agent decorator
from reminix-runtime.

Usage:
    python main.py

Then test the endpoints:

    # Health check
    curl http://localhost:8080/health

    # Discovery
    curl http://localhost:8080/manifest

    # Execute endpoint
    curl -X POST http://localhost:8080/agents/echo/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"message": "Hello!"}}'

    # Response: {"output": "Echo: Hello!"}

    # Execute with messages (chat-style)
    curl -X POST http://localhost:8080/agents/echo/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'

    # Response: {"output": "You said: Hello!"}

    # Streaming execute
    curl -X POST http://localhost:8080/agents/echo/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"message": "Hello!"}, "stream": true}'
"""

import json

from reminix_runtime import agent, serve


@agent(name="echo")
async def echo(
    message: str = "",
    messages: list | None = None,
    context: dict | None = None,
):
    """A simple echo agent that demonstrates the decorator pattern.

    Supports both task-style (message) and chat-style (messages) requests,
    with optional streaming.

    Args:
        message: A message to echo back
        messages: Chat-style messages list
        context: Optional request context
    """
    # Check if this is a chat-style request (has messages)
    if messages and isinstance(messages, list):
        user_message = messages[-1]["content"] if messages else ""
        response = f"You said: {user_message}"
    else:
        # Access optional context
        user_id = None
        if context:
            user_id = context.get("user_id")

        response = f"Echo: {message}"
        if user_id:
            response += f" (from user {user_id})"

    # Stream the response word by word
    words = response.split()
    for i, word in enumerate(words):
        chunk = word if i == 0 else f" {word}"
        yield json.dumps({"chunk": chunk})


if __name__ == "__main__":
    print("Custom Agent Example")
    print("=" * 40)
    print(f"Agent: {echo.name}")
    print(f"Streaming: {echo.metadata.get('capabilities', {}).get('streaming', False)}")
    print()
    print("Server running on http://localhost:8080")
    print()
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /manifest")
    print("  POST /agents/echo/invoke")
    print()

    serve(agents=[echo])
