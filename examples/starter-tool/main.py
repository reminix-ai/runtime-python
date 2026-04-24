"""Starter Tool — the fastest way to build a tool on Reminix.

Defines a minimal MCP tool with the `@tool` decorator and serves it via MCP
Streamable HTTP. Tools are the atomic units agents call; Reminix exposes them
through a standard `/mcp` endpoint so any MCP client can discover and invoke
them.

Call via MCP: POST /mcp with a JSON-RPC `tools/call` request.
"""

from pydantic import BaseModel, Field

from reminix_runtime import serve, tool


class GreetOutput(BaseModel):
    greeting: str = Field(description="Generated greeting")


@tool(name="greet")
async def greet(name: str) -> GreetOutput:
    """Greet someone by name.

    Args:
        name: Name of the person to greet.
    """
    return GreetOutput(greeting=f"Hello, {name}!")


if __name__ == "__main__":
    serve(tools=[greet])
