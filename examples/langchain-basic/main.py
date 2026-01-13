"""
Basic LangChain agent example

This example shows how to create a simple LangChain agent
and serve it via Reminix Runtime.
"""

from reminix_runtime import serve
from reminix_langchain import wrap

# from langchain_openai import ChatOpenAI
# from langgraph.prebuilt import create_react_agent

# Example: Create a LangChain agent
# model = ChatOpenAI(model="gpt-4o")
# agent = create_react_agent(model, tools=[...])


# For now, we'll use a placeholder
class PlaceholderAgent:
    def invoke(self, input: dict) -> dict:
        return {"messages": [{"role": "assistant", "content": "Hello from LangChain!"}]}


agent = PlaceholderAgent()

# Wrap the agent with the Reminix adapter
wrapped_agent = wrap(agent, name="langchain-basic")

# Serve the agent
if __name__ == "__main__":
    print("Server running on http://localhost:8080")
    serve([wrapped_agent], port=8080)
