"""
Basic LangChain agent example

This example shows how to create a simple LangChain agent
and serve it via Reminix Runtime.

Usage:
    python main.py

Then test the endpoints:

    # Invoke endpoint (task-oriented)
    curl -X POST http://localhost:8080/agents/langchain-basic/invoke \
      -H "Content-Type: application/json" \
      -d '{"input": {"query": "What is AI?"}}'
    
    # Response: {"output": "..."}

    # Chat endpoint (conversational)
    curl -X POST http://localhost:8080/agents/langchain-basic/chat \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
    
    # Response: {"output": "...", "messages": [...]}
"""

from reminix_runtime import serve
from reminix_langchain import wrap

# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent

# Example: Create a LangChain agent
# model = ChatOpenAI(model="gpt-4o")
# agent = create_react_agent(model, tools=[...])


# For demo purposes, we'll use a placeholder that mimics a LangChain Runnable.
# In production, replace this with a real LangChain model or chain.
class PlaceholderRunnable:
    """A mock runnable that behaves like a LangChain model."""

    async def ainvoke(self, input: any) -> dict:
        # For invoke: receives arbitrary input, returns result
        if isinstance(input, dict):
            query = input.get("query", str(input))
            return {"content": f"You asked: {query}"}
        # For chat: receives list of messages, returns AI message
        elif isinstance(input, list):
            return {"content": "Hello from LangChain!"}
        return {"content": str(input)}


agent = PlaceholderRunnable()

# Wrap the agent with the Reminix adapter
wrapped_agent = wrap(agent, name="langchain-basic")

# Serve the agent
if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    print("\nEndpoints:")
    print("  GET  /health")
    print("  GET  /info")
    print("  POST /agents/langchain-basic/invoke")
    print("  POST /agents/langchain-basic/chat")
    serve([wrapped_agent], port=8080)
