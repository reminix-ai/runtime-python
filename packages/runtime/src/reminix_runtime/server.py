"""Reminix Runtime Server."""

from .adapters.base import BaseAdapter


def serve(agents: list[BaseAdapter], port: int = 8080, host: str = "0.0.0.0") -> None:
    """Serve agents via REST API.

    Args:
        agents: List of wrapped agents (adapters).
        port: Port to serve on.
        host: Host to bind to.
    """
    # TODO: Implement server
    raise NotImplementedError("serve() is not yet implemented")
