"""LangChain Agent — LangChain served through Reminix.

`LangChainChatAgent` adapts a LangChain chat model (or any Runnable) into a
Reminix-compatible agent. Use it to serve existing LangChain compositions as
a streaming REST API without rewriting them.

Invoke: POST /agents/langchain-agent/invoke with {"input": {"prompt": "..."}}
or {"input": {"messages": [{"role": "...", "content": "..."}]}}.
"""

from langchain_openai import ChatOpenAI

from reminix_langchain import LangChainChatAgent
from reminix_runtime import serve

agent = LangChainChatAgent(
    ChatOpenAI(model="gpt-4o-mini"),
    name="langchain-agent",
)

if __name__ == "__main__":
    serve(agents=[agent])
