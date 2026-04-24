"""Starter Agent — the fastest way to try Reminix.

Defines a minimal agent with the `@agent` decorator and serves it as a REST
API. No API keys, no external SDKs — just the runtime itself. Use this as a
starting point for your own agent.

Invoke: POST /agents/starter-agent/invoke with {"input": {"message": "..."}}.
"""

from reminix_runtime import agent, serve


@agent(name="starter-agent")
async def starter(message: str = "") -> str:
    """A minimal starter agent."""
    return f"You said: {message}"


if __name__ == "__main__":
    serve(agents=[starter])
