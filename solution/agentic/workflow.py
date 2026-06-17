from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from data.models.state import TicketState
from agentic.agents import supervisor, classifier, resolver, escalation

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

    return END


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

    return END


def build_graph() -> StateGraph:
    """
    Constructs and returns the compiled UDA-Hub LangGraph.

    Nodes:
        supervisor  — orchestrates routing (entry point)
        classifier  — classifies ticket and returns to supervisor
        resolver    — attempts resolution with tools and RAG
        escalation  — handles unresolvable tickets, briefs human agent

    Edges:
        START → supervisor
        supervisor →(conditional)→ classifier | resolver | escalation | END
        classifier → supervisor (always loops back after classification)
        resolver →(conditional)→ escalation | END
        escalation → END
    """
    graph = StateGraph(TicketState)

    # ── Register nodes ──
    graph.add_node("supervisor", supervisor.run)
    graph.add_node("classifier", classifier.run)
    graph.add_node("resolver", resolver.run)
    graph.add_node("escalation", escalation.run)


    graph.add_edge(START, "supervisor")
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

    graph.add_edge("classifier", "supervisor")
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