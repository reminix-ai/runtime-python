# Google Gemini Basic Example

A simple example showing how to create a Google Gemini chat agent with Reminix.

## Setup

1. Get a Google API key from [Google AI Studio](https://aistudio.google.com/)
2. Create a `.env` file in the repository root:
   ```
   GOOGLE_API_KEY=your-api-key
   ```

## Run

```bash
python main.py
```

## Test

```bash
# Simple prompt
curl -X POST http://localhost:8080/agents/google-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "What is the capital of France?"}}'

# Chat-style messages
curl -X POST http://localhost:8080/agents/google-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello!"}]}}'

# Streaming
curl -X POST http://localhost:8080/agents/google-basic/invoke \
  -H "Content-Type: application/json" \
  -d '{"input": {"prompt": "Tell me a story"}, "stream": true}'
```
