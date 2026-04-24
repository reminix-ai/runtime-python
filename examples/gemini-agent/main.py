"""Google Gemini Agent — Gemini served through Reminix.

`GoogleChatAgent` wraps a genai.Client so Reminix can invoke it through the
uniform Agent interface. The adapter translates Reminix invoke requests into
Gemini generate-content calls and streams responses back — you get tool use
and message-history handling without writing any protocol glue.

Invoke: POST /agents/gemini-agent/invoke with {"input": {"prompt": "..."}}
or {"input": {"messages": [{"role": "...", "content": "..."}]}}.
"""

from google import genai

from reminix_google import GoogleChatAgent
from reminix_runtime import serve

agent = GoogleChatAgent(
    genai.Client(),
    name="gemini-agent",
    model="gemini-2.5-flash",
)

if __name__ == "__main__":
    serve(agents=[agent])
