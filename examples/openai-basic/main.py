"""
Basic OpenAI example

This example shows how to create a simple OpenAI chat agent
and serve it via Reminix Runtime.

Requirements:
    pip install reminix-openai python-dotenv

Environment:
    Create a .env file in the repository root with:
    OPENAI_API_KEY=your-api-key

Usage:
    python main.py

Then test the endpoints:

    # With a simple prompt
    curl -X POST http://localhost:8080/agents/openai-basic/execute \
      -H "Content-Type: application/json" \
      -d '{"input": {"prompt": "What is the capital of France?"}}'

    # Response: {"output": "The capital of France is Paris."}

    # With messages (chat-style)
    curl -X POST http://localhost:8080/agents/openai-basic/execute \
      -H "Content-Type: application/json" \
      -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'

    # Response: {"output": "Hello! How can I help you today?"}
"""

from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from reminix_openai import wrap
from reminix_runtime import serve

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Create an OpenAI client
client = AsyncOpenAI()

# Wrap the client with the Reminix adapter
agent = wrap(client, name="openai-basic", model="gpt-4o-mini")

# Serve the agent
if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/openai-basic/execute")
    serve(agents=[agent], port=8080)
