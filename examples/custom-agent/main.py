"""
Custom Agent Example

This example shows how to create a custom agent using the decorator-based
Agent class from reminix-runtime.

Usage:
    python main.py

Then test the endpoints:

    # Health check
    curl http://localhost:8080/health

    # Discovery
    curl http://localhost:8080/info

    # Execute endpoint
    curl -X POST http://localhost:8080/agents/echo/execute \
      -H "Content-Type: application/json" \
      -d '{"input": {"message": "Hello!"}}'

    # Response: {"output": "Echo: Hello!"}

    # Execute with messages (chat-style)
    curl -X POST http://localhost:8080/agents/echo/execute \
      -H "Content-Type: application/json" \
      -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'

    # Response: {"output": "You said: Hello!"}

    # Streaming execute
    curl -X POST http://localhost:8080/agents/echo/execute \
      -H "Content-Type: application/json" \
      -d '{"input": {"message": "Hello!"}, "stream": true}'
"""

import json

from reminix_runtime import (
    Agent,
    ExecuteRequest,
    ExecuteResponse,
    serve,
)

# Create an agent with metadata
agent = Agent(
    "echo",
    metadata={
        "description": "A simple echo agent that demonstrates the decorator pattern",
        "version": "1.0.0",
    },
)


@agent.on_execute
async def handle_execute(request: ExecuteRequest) -> ExecuteResponse:
    """Handle execute requests."""
    # Check if this is a chat-style request (has messages)
    if "messages" in request.input and isinstance(request.input["messages"], list):
        messages = request.input["messages"]
        user_message = messages[-1]["content"] if messages else ""
        return ExecuteResponse(output=f"You said: {user_message}")

    # Otherwise treat as task-style request
    message = request.input.get("message", "")

    # Access optional context
    user_id = None
    if request.context:
        user_id = request.context.get("user_id")

    output = f"Echo: {message}"
    if user_id:
        output += f" (from user {user_id})"

    return ExecuteResponse(output=output)


@agent.on_execute_stream
async def handle_execute_stream(request: ExecuteRequest):
    """Handle streaming execute requests."""
    # Check if this is a chat-style request (has messages)
    if "messages" in request.input and isinstance(request.input["messages"], list):
        messages = request.input["messages"]
        user_message = messages[-1]["content"] if messages else ""
        response = f"You said: {user_message}"
    else:
        message = request.input.get("message", "")
        response = f"Echo: {message}"

    # Stream the response word by word
    words = response.split()
    for i, word in enumerate(words):
        chunk = word if i == 0 else f" {word}"
        yield json.dumps({"chunk": chunk})


if __name__ == "__main__":
    print("Custom Agent Example")
    print("=" * 40)
    print(f"Agent: {agent.name}")
    print(f"Streaming: {agent.streaming}")
    print()
    print("Server running on http://localhost:8080")
    print()
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/echo/execute")
    print()

    serve(agents=[agent], port=8080)
