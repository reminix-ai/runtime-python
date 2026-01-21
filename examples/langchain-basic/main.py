"""
Basic LangChain agent example

This example shows how to create a simple LangChain agent
and serve it via Reminix Runtime.

Requirements:
    pip install reminix-langchain langchain-openai python-dotenv

Environment:
    Create a .env file in the repository root with:
    OPENAI_API_KEY=your-api-key

Usage:
    python main.py

Then test the endpoints:

    # Invoke endpoint (task-oriented)
    curl -X POST http://localhost:8080/agents/langchain-basic/execute \
      -H "Content-Type: application/json" \
      -d '{"input": {"prompt": "What is AI?"}}'

    # Response: {"output": "AI (Artificial Intelligence) is..."}

    # Chat endpoint (conversational)
    curl -X POST http://localhost:8080/agents/langchain-basic/execute \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

    # Response: {"output": "Hello! How can I help you today?", "messages": [...]}
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from reminix_langchain import wrap
from reminix_runtime import serve

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Create a LangChain chat model
model = ChatOpenAI(model="gpt-4o-mini")

# Wrap the model with the Reminix adapter
agent = wrap(model, name="langchain-basic")

# Serve the agent
if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/langchain-basic/execute")
    serve(agents=[agent], port=8080)
