"""
LlamaIndex RAG example

This example shows how to create a LlamaIndex ReAct agent with tools
and serve it via Reminix Runtime.

Requirements:
    pip install reminix-llamaindex llama-index-llms-openai python-dotenv

Environment:
    Create a .env file in the repository root with:
    OPENAI_API_KEY=your-api-key

Usage:
    python main.py

Then test the endpoints:

    # Invoke endpoint (task-oriented)
    curl -X POST http://localhost:8080/agents/llamaindex-rag/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"query": "What is the weather in Paris?"}}'

    # Response: {"output": "The weather in Paris is sunny with a temperature of 22°C."}

    # Chat endpoint (conversational)
    curl -X POST http://localhost:8080/agents/llamaindex-rag/chat \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}'

    # Response: {"output": "The weather in Tokyo is rainy with a temperature of 18°C.", "messages": [...]}
"""

from pathlib import Path

from dotenv import load_dotenv
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI

from reminix_llamaindex import wrap
from reminix_runtime import serve

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# Define a tool for the agent to use
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Partly cloudy, 20°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


class ChatEngineWrapper:
    """Wrapper to adapt LlamaIndex ReActAgent workflow to ChatEngine interface."""

    def __init__(self, agent: ReActAgent):
        self._agent = agent
        self._ctx = Context(agent)

    async def achat(self, message: str):
        """Async chat method compatible with LlamaIndex ChatEngine protocol."""
        handler = self._agent.run(message, ctx=self._ctx)
        return await handler

    async def astream_chat(self, message: str):
        """Async streaming chat - not implemented."""
        raise NotImplementedError("Streaming not implemented for workflow agents")


# Create a LlamaIndex ReAct agent with tools
llm = OpenAI(model="gpt-4o-mini")
react_agent = ReActAgent(tools=[get_weather], llm=llm)
engine = ChatEngineWrapper(react_agent)

# Wrap the engine with the Reminix adapter
agent = wrap(engine, name="llamaindex-rag")

# Serve the agent
if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/llamaindex-rag/invoke")
    print("  POST /agents/llamaindex-rag/chat")
    print("\nAvailable tools:")
    print("  - get_weather(city): Get weather for Paris, London, Tokyo, or New York")
    serve(agents=[agent], port=8080)
