"""LangGraph workflow adapter for Reminix Runtime."""

from typing import Any

from langgraph.errors import GraphInterrupt
from langgraph.types import Command

from reminix_runtime import AGENT_TEMPLATES, AgentRequest


class LangGraphWorkflowAgent:
    """Workflow agent adapter for LangGraph compiled graphs.

    Maps LangGraph's streaming per-node outputs and interrupt/resume
    to the workflow template's {status, steps, result, pendingAction} output schema.
    """

    def __init__(self, graph: Any, name: str = "langgraph-workflow-agent") -> None:
        self._graph = graph
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "description": "langgraph workflow adapter",
            "capabilities": {"streaming": False},
            "input": AGENT_TEMPLATES["workflow"]["input"],
            "output": AGENT_TEMPLATES["workflow"]["output"],
            "adapter": "langgraph",
            "template": "workflow",
        }

    async def invoke(self, request: AgentRequest) -> dict[str, Any]:
        # 1. Extract thread_id from request.context for checkpointed graphs
        config: dict[str, Any] = {}
        if request.context and "thread_id" in request.context:
            config = {"configurable": {"thread_id": request.context["thread_id"]}}

        # 2. Determine input: resume vs normal
        if "resume" in request.input and request.input["resume"] is not None:
            resume_data = request.input["resume"]
            graph_input = Command(resume=resume_data.get("input"))
        else:
            graph_input = request.input

        # 3. Stream graph and collect steps
        steps: list[dict[str, Any]] = []
        last_node: str | None = None

        try:
            async for chunk in self._graph.astream(graph_input, config):
                if isinstance(chunk, dict):
                    for node_name, node_output in chunk.items():
                        last_node = node_name
                        steps.append(
                            {
                                "name": node_name,
                                "status": "completed",
                                "output": node_output,
                            }
                        )

        except GraphInterrupt as exc:
            # 4. Handle interrupts
            # GraphInterrupt stores interrupts in args[0] (a sequence of Interrupt)
            interrupts = exc.args[0] if exc.args else []
            interrupt = interrupts[0] if interrupts else None
            interrupt_value = interrupt.value if interrupt else None

            pending_action: dict[str, Any]
            if (
                isinstance(interrupt_value, dict)
                and "type" in interrupt_value
                and "message" in interrupt_value
            ):
                pending_action = {
                    "step": interrupt_value.get("step", last_node or "unknown"),
                    "type": interrupt_value["type"],
                    "message": interrupt_value["message"],
                }
                if "options" in interrupt_value:
                    pending_action["options"] = interrupt_value["options"]
            elif isinstance(interrupt_value, str):
                pending_action = {
                    "step": last_node or "unknown",
                    "type": "input",
                    "message": interrupt_value,
                }
            else:
                pending_action = {
                    "step": last_node or "unknown",
                    "type": "input",
                    "message": str(interrupt_value),
                }

            # Mark last step as paused
            if steps:
                steps[-1]["status"] = "paused"

            return {
                "output": {
                    "status": "paused",
                    "steps": steps,
                    "pendingAction": pending_action,
                },
            }

        except Exception as exc:
            # 5. Handle errors
            if steps:
                steps[-1]["status"] = "failed"

            return {
                "output": {
                    "status": "failed",
                    "steps": steps,
                    "error": str(exc),
                },
            }

        # 6. Normal completion
        result = steps[-1]["output"] if steps else None

        output: dict[str, Any] = {
            "status": "completed",
            "steps": steps,
        }
        if result is not None:
            output["result"] = result

        return {"output": output}
