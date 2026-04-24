"""OpenAI SDK Agent — GPT served through Reminix.

`OpenAIChatAgent` wraps an OpenAI SDK client so Reminix can invoke it through
a uniform agent interface. You get streaming, function calling, and
message-history handling without writing protocol glue.

Invoke: POST /agents/openai-agent/invoke with {"input": {"prompt": "..."}}
or {"input": {"messages": [{"role": "...", "content": "..."}]}}.
"""

from openai import AsyncOpenAI

from reminix_openai import OpenAIChatAgent
from reminix_runtime import serve

agent = OpenAIChatAgent(
    AsyncOpenAI(),
    name="openai-agent",
    model="gpt-4o-mini",
)

if __name__ == "__main__":
    serve(agents=[agent])
