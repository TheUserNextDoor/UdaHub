from typing import Any, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage

from data.models.state import TicketState
from agentic.agents import supervisor, classifier, resolver, escalation
from agentic.tools.ticket_tools import get_long_term_memory


def _build_prior_turns(messages: list[Any], max_lines: int = 8) -> str:
    """Builds a compact conversation history block for short-term memory."""
    history_lines: list[str] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history_lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_lines.append(f"Assistant: {msg.content}")

    if len(history_lines) > max_lines:
        history_lines = history_lines[-max_lines:]

    return "\n".join(history_lines)


async def hydrate_memory(state: TicketState) -> dict:
    """Hydrates short-term and long-term memory before supervisor routing."""
    messages = state.get("messages", [])
    short_term_memory = {
        "prior_turns": _build_prior_turns(messages),
    }

    long_term_memory = state.get("long_term_memory") or {
        "past_resolutions": [],
        "past_issue_types": [],
        "preferences": {},
    }

    ticket = state.get("ticket", {})
    external_user_id = ticket.get("external_user_id")
    if external_user_id:
        try:
            loaded_memory = await get_long_term_memory(external_user_id)
            if isinstance(loaded_memory, dict):
                long_term_memory = {
                    "past_resolutions": loaded_memory.get("past_resolutions", []),
                    "past_issue_types": loaded_memory.get("past_issue_types", []),
                    "preferences": loaded_memory.get("preferences", {}),
                }
        except Exception:
            # Keep workflow resilient if memory service is temporarily unavailable.
            pass

    return {
        "short_term_memory": short_term_memory,
        "long_term_memory": long_term_memory,
    }

def route_from_supervisor(
    state: TicketState,
) -> Literal["classifier", "resolver", "escalation", "__end__"]:
    """
    Conditional edge from the Supervisor node.
    Reads next_agent from state and returns the next node name.
    """
    next_agent = state.get("next_agent")

    if next_agent == "classifier":
        return "classifier"
    if next_agent == "resolver":
        return "resolver"
    if next_agent == "escalation":
        return "escalation"

    return "__end__"


def route_from_resolver(
    state: TicketState,
) -> Literal["escalation", "__end__"]:
    """
    Conditional edge from the Resolver node.
    If resolved=False, escalate. Otherwise end.
    """
    next_agent = state.get("next_agent")

    if next_agent == "escalation":
        return "escalation"

    return "__end__"


def build_graph() -> StateGraph:
    """
    Constructs and returns the compiled UDA-Hub LangGraph.

    Nodes:
        memory      — hydrates short-term and long-term memory
        supervisor  — orchestrates routing (entry point)
        classifier  — classifies ticket and returns to supervisor
        resolver    — attempts resolution with tools and RAG
        escalation  — handles unresolvable tickets, briefs human agent

    Edges:
        START → memory → supervisor
        supervisor →(conditional)→ classifier | resolver | escalation | END
        classifier → memory → supervisor (refreshes memory before routing)
        resolver →(conditional)→ escalation | END
        escalation → END
    """
    graph = StateGraph(TicketState)

    # ── Register nodes ──
    graph.add_node("memory", hydrate_memory)
    graph.add_node("supervisor", supervisor.run)
    graph.add_node("classifier", classifier.run)
    graph.add_node("resolver", resolver.run)
    graph.add_node("escalation", escalation.run)

    graph.add_edge(START, "memory")
    graph.add_edge("memory", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "classifier": "classifier",
            "resolver": "resolver",
            "escalation": "escalation",
            END: END,
        },
    )

    graph.add_edge("classifier", "memory")
    graph.add_conditional_edges(
        "resolver",
        route_from_resolver,
        {
            "escalation": "escalation",
            END: END,
        },
    )

    graph.add_edge("escalation", END)

    return graph

memory = MemorySaver()
graph = build_graph()
orchestrator = graph.compile(checkpointer=memory)