"""Graph assembly: wire the agents into a LangGraph ``StateGraph``.

Topology::

    START -> planner -> researcher -> approval_gate --(approved)--> writer
                                            |                         |
                                        (rejected)                 critic
                                            v                    /      \\
                                           END <-----(accept)---+  (revise, bounded)
                                                                          |
                                                                        writer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent_researcher.config import Settings
from agent_researcher.llm import create_chat_model
from agent_researcher.nodes import ResearchNodes
from agent_researcher.search import create_search_tool
from agent_researcher.state import ResearchState

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

    from agent_researcher.llm import ChatModel
    from agent_researcher.search import SearchTool


def build_graph(
    model: ChatModel | None = None,
    search_tool: SearchTool | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Compile the research pipeline into a runnable LangGraph app.

    Args:
        model: Chat model; defaults to the provider chosen in ``settings``.
        search_tool: Search backend; defaults likewise.
        settings: Pipeline configuration; defaults to environment values.
        checkpointer: Checkpoint saver; defaults to an in-memory saver,
            which enables thread-based resume and the approval interrupt.

    Returns:
        The compiled graph, ready for ``invoke`` / ``stream`` calls with a
        ``configurable.thread_id``.
    """
    settings = settings or Settings()
    model = model or create_chat_model(settings)
    search_tool = search_tool or create_search_tool(settings)
    nodes = ResearchNodes(model=model, search_tool=search_tool, settings=settings)

    graph = StateGraph(ResearchState)
    graph.add_node("planner", nodes.plan)
    graph.add_node("researcher", nodes.research)
    graph.add_node("approval_gate", nodes.approve)
    graph.add_node("writer", nodes.write)
    graph.add_node("critic", nodes.critique)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "approval_gate")
    graph.add_conditional_edges(
        "approval_gate",
        nodes.route_after_approval,
        {"write": "writer", "end": END},
    )
    graph.add_edge("writer", "critic")
    graph.add_conditional_edges(
        "critic",
        nodes.route_after_critic,
        {"revise": "writer", "accept": END},
    )
    return graph.compile(checkpointer=checkpointer or MemorySaver())
