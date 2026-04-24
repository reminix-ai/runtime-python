"""Anthropic SDK Agent — Claude served through Reminix.

`AnthropicChatAgent` wraps an Anthropic SDK client so Reminix can invoke it
through a uniform agent interface. You get streaming, tool use, and
message-history handling without writing protocol glue.

Invoke: POST /agents/anthropic-agent/invoke with {"input": {"prompt": "..."}}
or {"input": {"messages": [{"role": "...", "content": "..."}]}}.
"""

from anthropic import AsyncAnthropic

from reminix_anthropic import AnthropicChatAgent
from reminix_runtime import serve

agent = AnthropicChatAgent(
    AsyncAnthropic(),
    name="anthropic-agent",
    model="claude-sonnet-4-6",
)

if __name__ == "__main__":
    serve(agents=[agent])
