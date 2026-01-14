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

    # Invoke endpoint (task-oriented)
    curl -X POST http://localhost:8080/agents/echo/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"message": "Hello!"}}'

    # Response: {"output": "Echo: Hello!"}

    # Chat endpoint (conversational)
    curl -X POST http://localhost:8080/agents/echo/chat \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

    # Response: {"output": "You said: Hello!", "messages": [...]}

    # Streaming invoke
    curl -X POST http://localhost:8080/agents/echo/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"message": "Hello!"}, "stream": true}'
"""

import json

from reminix_runtime import (
    Agent,
    ChatRequest,
    ChatResponse,
    InvokeRequest,
    InvokeResponse,
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


@agent.on_invoke
async def handle_invoke(request: InvokeRequest) -> InvokeResponse:
    """Handle invoke requests - task-oriented operations."""
    message = request.input.get("message", "")

    # Access optional context
    user_id = None
    if request.context:
        user_id = request.context.get("user_id")

    output = f"Echo: {message}"
    if user_id:
        output += f" (from user {user_id})"

    return InvokeResponse(output=output)


@agent.on_chat
async def handle_chat(request: ChatRequest) -> ChatResponse:
    """Handle chat requests - conversational interactions."""
    # Get the last user message
    user_message = request.messages[-1].content if request.messages else ""

    response = f"You said: {user_message}"

    return ChatResponse(
        output=response,
        messages=[
            *[{"role": m.role, "content": m.content} for m in request.messages],
            {"role": "assistant", "content": response},
        ],
    )


@agent.on_invoke_stream
async def handle_invoke_stream(request: InvokeRequest):
    """Handle streaming invoke requests."""
    message = request.input.get("message", "")

    # Stream the response word by word
    words = f"Echo: {message}".split()
    for i, word in enumerate(words):
        chunk = word if i == 0 else f" {word}"
        yield json.dumps({"chunk": chunk})


@agent.on_chat_stream
async def handle_chat_stream(request: ChatRequest):
    """Handle streaming chat requests."""
    user_message = request.messages[-1].content if request.messages else ""

    # Stream the response word by word
    words = f"You said: {user_message}".split()
    for i, word in enumerate(words):
        chunk = word if i == 0 else f" {word}"
        yield json.dumps({"chunk": chunk})


if __name__ == "__main__":
    print("Custom Agent Example")
    print("=" * 40)
    print(f"Agent: {agent.name}")
    print(f"Invoke streaming: {agent.invoke_streaming}")
    print(f"Chat streaming: {agent.chat_streaming}")
    print()
    print("Server running on http://localhost:8080")
    print()
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/echo/invoke")
    print("  POST /agents/echo/chat")
    print()

    serve([agent], port=8080)
