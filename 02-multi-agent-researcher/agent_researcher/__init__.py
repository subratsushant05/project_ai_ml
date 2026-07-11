"""Multi-agent research assistant built on LangGraph.

A Planner -> Researcher -> Writer -> Critic pipeline with a bounded
revision loop, an optional human approval gate, and offline-first
model/search providers so everything runs deterministically without
network access or API keys.

Public names are imported lazily (PEP 562) so ``import agent_researcher``
stays lightweight and side-effect free.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_researcher.config import Settings as Settings
    from agent_researcher.graph import build_graph as build_graph
    from agent_researcher.state import Citation as Citation
    from agent_researcher.state import Critique as Critique
    from agent_researcher.state import Finding as Finding
    from agent_researcher.state import ResearchState as ResearchState

__version__ = "1.0.0"

_EXPORTS: dict[str, str] = {
    "Settings": "agent_researcher.config",
    "build_graph": "agent_researcher.graph",
    "Citation": "agent_researcher.state",
    "Critique": "agent_researcher.state",
    "Finding": "agent_researcher.state",
    "ResearchState": "agent_researcher.state",
}

__all__ = [*sorted(_EXPORTS), "__version__"]


def __getattr__(name: str) -> Any:
    """Resolve public names on first access."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(module_name), name)
