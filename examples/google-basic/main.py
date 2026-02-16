"""
Basic Google Gemini example

This example shows how to create a simple Google Gemini agent
and serve it via Reminix Runtime.

Requirements:
    pip install reminix-google python-dotenv

Environment:
    Create a .env file in the repository root with:
    GOOGLE_API_KEY=your-api-key

Usage:
    python main.py

Then test the endpoints:

    # With a simple prompt
    curl -X POST http://localhost:8080/agents/google-basic/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"prompt": "What is the capital of France?"}}'

    # Response: {"output": "The capital of France is Paris."}

    # With messages (chat-style)
    curl -X POST http://localhost:8080/agents/google-basic/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'

    # Response: {"output": "Hello! How can I help you today?"}
"""

from pathlib import Path

from dotenv import load_dotenv
from google import genai

from reminix_google import GoogleChatAgent
from reminix_runtime import serve

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Create a Google GenAI client
client = genai.Client()

# Create and serve the agent
agent = GoogleChatAgent(client, name="google-basic", model="gemini-2.5-flash")

if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /manifest")
    print("  POST /agents/google-basic/invoke")
    serve(agents=[agent])
