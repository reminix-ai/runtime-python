"""
Basic Anthropic example

This example shows how to create a simple Anthropic Claude agent
and serve it via Reminix Runtime.

Requirements:
    pip install reminix-anthropic python-dotenv

Environment:
    Create a .env file in the repository root with:
    ANTHROPIC_API_KEY=your-api-key

Usage:
    python main.py

Then test the endpoints:

    # Invoke endpoint (task-oriented)
    curl -X POST http://localhost:8080/agents/anthropic-basic/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"prompt": "What is the capital of France?"}}'

    # Response: {"output": "The capital of France is Paris."}

    # Chat endpoint (conversational)
    curl -X POST http://localhost:8080/agents/anthropic-basic/chat \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

    # Response: {"output": "Hello! How can I help you today?", "messages": [...]}
"""

from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv

from reminix_anthropic import wrap
from reminix_runtime import serve

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Create an Anthropic client
client = AsyncAnthropic()

# Wrap the client with the Reminix adapter
agent = wrap(client, name="anthropic-basic", model="claude-3-haiku-20240307")

# Serve the agent
if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/anthropic-basic/invoke")
    print("  POST /agents/anthropic-basic/chat")
    serve([agent], port=8080)
