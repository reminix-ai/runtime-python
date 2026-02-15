"""
LangGraph with Tools example

This example shows how to create a LangGraph ReAct agent with tool calling
and serve it via Reminix Runtime.

Requirements:
    pip install reminix-langgraph langchain-openai langgraph python-dotenv

Environment:
    Create a .env file in the repository root with:
    OPENAI_API_KEY=your-api-key

Usage:
    python main.py

Then test the endpoints:

    # Invoke endpoint (task-oriented)
    curl -X POST http://localhost:8080/agents/langgraph-tools/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"messages": [{"role": "user", "content": "What is the weather in Paris?"}]}}'

    # Response: {"output": "The weather in Paris is sunny with a temperature of 22°C."}

    # Chat endpoint (conversational)
    curl -X POST http://localhost:8080/agents/langgraph-tools/invoke \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "What is the weather in Tokyo?"}]}'

    # Response: {"output": "The weather in Tokyo is rainy with a temperature of 18°C.", "messages": [...]}
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from reminix_langgraph import wrap
from reminix_runtime import serve

# Load environment variables from root .env file
load_dotenv(Path(__file__).parent.parent.parent / ".env")


# Define a tool for the agent to use
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "paris": "Sunny, 22°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Partly cloudy, 20°C",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


# Create a LangGraph ReAct agent with tools
llm = ChatOpenAI(model="gpt-4o-mini")
graph = create_react_agent(llm, tools=[get_weather])

# Wrap the graph with the Reminix adapter
agent = wrap(graph, name="langgraph-tools")

# Serve the agent
if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/langgraph-tools/invoke")
    print("\nAvailable tools:")
    print("  - get_weather(city): Get weather for Paris, London, Tokyo, or New York")
    serve(agents=[agent])
