__version__ = "0.0.19"

from .agent import AgentLike, RuntimeAgent, agent
from .content import build_messages_from_input, message_content_to_text
from .schemas import (
    ADAPTER_INPUT,
    AGENT_TEMPLATES,
    CONTENT_PART_SCHEMA,
    DEFAULT_AGENT_INPUT,
    DEFAULT_AGENT_OUTPUT,
    DEFAULT_AGENT_TEMPLATE,
    MESSAGE_SCHEMA,
    TOOL_CALL_SCHEMA,
    AgentTemplate,
)
from .server import create_app, serve
from .tool import Tool, ToolLike, tool
from .types import (
    AgentRequest,
    AgentResponse,
    Capabilities,
    Message,
    Role,
    RuntimeError,
    RuntimeErrorResponse,
    ToolCall,
    ToolRequest,
    ToolResponse,
)

__all__ = [
    "__version__",
    "serve",
    "create_app",
    "message_content_to_text",
    "build_messages_from_input",
    # Schemas
    "ADAPTER_INPUT",
    "AgentTemplate",
    "AGENT_TEMPLATES",
    "DEFAULT_AGENT_TEMPLATE",
    "DEFAULT_AGENT_INPUT",
    "DEFAULT_AGENT_OUTPUT",
    "TOOL_CALL_SCHEMA",
    "CONTENT_PART_SCHEMA",
    "MESSAGE_SCHEMA",
    # Base types
    "Role",
    "Message",
    "ToolCall",
    "Capabilities",
    "RuntimeError",
    "RuntimeErrorResponse",
    # Agent
    "AgentRequest",
    "AgentResponse",
    "AgentLike",
    "RuntimeAgent",
    "agent",
    # Tool
    "ToolRequest",
    "ToolResponse",
    "ToolLike",
    "Tool",
    "tool",
]
