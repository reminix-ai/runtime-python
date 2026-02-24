"""Agent type schemas for Reminix Runtime.

Defines named types with predefined input/output JSON schemas.
These types standardize common agent patterns (prompt, chat, task, etc.)
so that clients and tooling can interoperate without inspecting individual schemas.
"""

from typing import Any, Literal

AgentType = Literal["prompt", "chat", "task", "rag", "thread", "workflow"]

DEFAULT_AGENT_TYPE: AgentType = "prompt"

# JSON schema for a single tool call (OpenAI-style)
TOOL_CALL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "Tool call id"},
        "type": {"type": "string", "enum": ["function"], "description": "Tool call type"},
        "function": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Function/tool name"},
                "arguments": {"type": "string", "description": "JSON string of arguments"},
            },
            "required": ["name", "arguments"],
        },
    },
    "required": ["id", "type", "function"],
}

# Content part schema (text, image_url, input_audio, file, refusal)
CONTENT_PART_SCHEMA: dict[str, Any] = {
    "oneOf": [
        {
            "type": "object",
            "properties": {"type": {"const": "text"}, "text": {"type": "string"}},
            "required": ["type", "text"],
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "image_url"},
                "image_url": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "detail": {"type": "string", "enum": ["auto", "low", "high"]},
                    },
                    "required": ["url"],
                },
            },
            "required": ["type", "image_url"],
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "input_audio"},
                "input_audio": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string", "description": "Base64 encoded audio data"},
                        "format": {"type": "string", "enum": ["wav", "mp3"]},
                    },
                    "required": ["data", "format"],
                },
            },
            "required": ["type", "input_audio"],
        },
        {
            "type": "object",
            "properties": {
                "type": {"const": "file"},
                "file": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "filename": {"type": "string"},
                        "file_data": {"type": "string", "description": "Base64 encoded file data"},
                    },
                },
            },
            "required": ["type", "file"],
        },
        {
            "type": "object",
            "properties": {"type": {"const": "refusal"}, "refusal": {"type": "string"}},
            "required": ["type", "refusal"],
        },
    ],
}

# JSON schema for a message (OpenAI-style; input and output)
MESSAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "role": {
            "type": "string",
            "enum": ["system", "user", "assistant", "tool"],
            "description": "Message role",
        },
        "content": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": CONTENT_PART_SCHEMA, "minItems": 1},
                {"type": "null"},
            ],
            "description": "Message content: string, array of content parts, or null when tool_calls present",
        },
        "name": {"type": "string", "description": "Optional participant name"},
        "tool_call_id": {
            "type": "string",
            "description": 'Tool call ID (required when role is "tool")',
        },
        "tool_calls": {
            "type": "array",
            "description": "Tool calls (assistant role only)",
            "items": TOOL_CALL_SCHEMA,
        },
    },
}

AGENT_TYPES: dict[AgentType, dict[str, Any]] = {
    "prompt": {
        "inputSchema": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The prompt or task for the agent"},
            },
            "required": ["prompt"],
        },
        "outputSchema": {"type": "string"},
    },
    "chat": {
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Chat messages (OpenAI-style)",
                    "items": MESSAGE_SCHEMA,
                },
            },
            "required": ["messages"],
        },
        "outputSchema": {"type": "string"},
    },
    "task": {
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Task name or description for stateless, single-shot execution",
                },
            },
            "required": ["task"],
            "additionalProperties": True,
        },
        "outputSchema": {
            "description": "Structured JSON result of stateless, single-shot execution",
            "type": "object",
            "additionalProperties": True,
        },
    },
    "rag": {
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The question to answer from documents"},
                "messages": {
                    "type": "array",
                    "description": "Optional prior conversation (chat-style RAG)",
                    "items": MESSAGE_SCHEMA,
                },
                "collectionIds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional knowledge collection IDs to scope the search",
                },
            },
            "required": ["query"],
        },
        "outputSchema": {"type": "string"},
    },
    "thread": {
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Chat messages with tool_calls and tool results (OpenAI-style)",
                    "items": MESSAGE_SCHEMA,
                },
            },
            "required": ["messages"],
        },
        "outputSchema": {
            "type": "array",
            "description": "Updated message thread (OpenAI-style, may include assistant message and tool_calls)",
            "items": MESSAGE_SCHEMA,
        },
    },
    "workflow": {
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Workflow task or description",
                },
                "state": {
                    "type": "object",
                    "description": "Previous execution output passed back on resume (steps, result, etc.)",
                    "additionalProperties": True,
                },
                "resume": {
                    "type": "object",
                    "description": "Resume a paused workflow from a specific step",
                    "properties": {
                        "step": {
                            "type": "string",
                            "description": "Step name to resume from",
                        },
                        "input": {
                            "type": "object",
                            "description": "Optional input for the resumed step",
                            "additionalProperties": True,
                        },
                    },
                    "required": ["step"],
                },
            },
            "required": ["task"],
            "additionalProperties": True,
        },
        "outputSchema": {
            "type": "object",
            "description": "Workflow execution result with step-level status tracking",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["completed", "failed", "paused", "running"],
                    "description": "Overall workflow status",
                },
                "steps": {
                    "type": "array",
                    "description": "Step-level execution results",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "Step name"},
                            "status": {
                                "type": "string",
                                "enum": ["completed", "failed", "paused", "skipped", "pending"],
                                "description": "Step execution status",
                            },
                            "output": {"description": "Step output (any JSON value)"},
                            "error": {
                                "type": "string",
                                "description": "Error message when step status is failed",
                            },
                        },
                        "required": ["name", "status"],
                    },
                },
                "result": {
                    "type": "object",
                    "description": "Final workflow result",
                    "additionalProperties": True,
                },
                "error": {
                    "type": "string",
                    "description": "Error message (when status is failed)",
                },
                "pendingAction": {
                    "type": "object",
                    "description": "Action required to continue (present when status is paused)",
                    "properties": {
                        "step": {
                            "type": "string",
                            "description": "Step awaiting action",
                        },
                        "type": {
                            "type": "string",
                            "description": "Action type (e.g. approval, input, decision)",
                        },
                        "message": {
                            "type": "string",
                            "description": "Human-readable description of the required action",
                        },
                        "options": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Available choices (for decision-type actions)",
                        },
                        "inputSchema": {
                            "type": "object",
                            "description": "JSON Schema describing expected input (for type: input actions)",
                            "additionalProperties": True,
                        },
                        "assignee": {
                            "type": "string",
                            "description": "Who should handle this action (email, user ID, or role)",
                        },
                    },
                    "required": ["step", "type", "message"],
                },
            },
            "required": ["status", "steps"],
        },
    },
}

# Default input/output schemas (same as prompt type)
DEFAULT_AGENT_INPUT: dict[str, Any] = AGENT_TYPES[DEFAULT_AGENT_TYPE]["inputSchema"]
DEFAULT_AGENT_OUTPUT: dict[str, Any] = AGENT_TYPES[DEFAULT_AGENT_TYPE]["outputSchema"]
