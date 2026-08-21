"""
Research loop for autonomous W2S experiments.

Provides the agent loop, base agent, and MCP tools for the autonomous research system.
"""

__all__ = [
    "BaseAgent",
    "AgentResult",
    "AutonomousAgentLoop",
]


def __getattr__(name):
    """Lazy imports to avoid heavy dependencies at module load time."""
    if name == "BaseAgent":
        from .agent import BaseAgent
        return BaseAgent
    elif name == "AgentResult":
        from .agent import AgentResult
        return AgentResult
    elif name == "AutonomousAgentLoop":
        from .agent import AutonomousAgentLoop
        return AutonomousAgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
