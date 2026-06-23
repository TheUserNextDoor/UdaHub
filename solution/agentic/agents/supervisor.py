
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from textwrap import dedent
import os
from data.models.state import TicketState, Classification


CONFIDENCE_THRESHOLD = 0.6

ESCALATION_ISSUE_TYPES = {
    "legal",
    "abuse",
    "fraud",
    "data_breach",
}

llm = ChatOpenAI(model="gpt-4o-mini",
                temperature=0,
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("VOCAREUM_KEY")
                )


FIRST_PASS_SYSTEM_PROMPT = dedent(
    """
    You are the Supervisor of UDA-Hub, an intelligent customer support system for CultPass,
    an experiences and subscription platform.

    Your role is to assess incoming support tickets and prepare them for classification.
    Be concise. Do not attempt to resolve the ticket yourself.
    Acknowledge the ticket and confirm it is being routed to the Classifier.
    """
).strip()

SECOND_PASS_SYSTEM_PROMPT = dedent(
    """
    You are the Supervisor of UDA-Hub. The Classifier has returned its assessment.
    Based on the classification, confirm the routing decision clearly and briefly.
    """
).strip()


def route(state: TicketState) -> str:
    """
    Determines the next node after the Supervisor has processed the state.

    Returns:
        "classifier"  — ticket not yet classified
        "resolver"    — confident enough to attempt resolution
        "escalation"  — low confidence or sensitive issue type
        "end"         — ticket already resolved
    """
    resolution = state.get("resolution")
    if resolution and resolution.get("resolved"):
        return "end"

    classification: Classification | None = state.get("classification")
    if not classification:
        return "classifier"

    if classification["issue_type"] in ESCALATION_ISSUE_TYPES:
        return "escalation"

    if classification["confidence"] < CONFIDENCE_THRESHOLD:
        return "escalation"

    return "resolver"


async def run(state: TicketState) -> dict:
    """
    Supervisor node function (async).

    On first pass (no classification yet): primes the state with a
    brief LLM assessment and sets next_agent = "classifier".

    On second pass (after classification): evaluates the classification
    and sets next_agent = "resolver" | "escalation".

    Returns:
        Partial state update dict (LangGraph merges this into the full state).
    """
    ticket = state["ticket"]
    classification = state.get("classification")


    if not classification:
        system_prompt = SystemMessage(content=FIRST_PASS_SYSTEM_PROMPT)
        human_message = HumanMessage(
            content=dedent(
                f"""
                New support ticket received.

                Ticket ID: {ticket['ticket_id']}
                Channel: {ticket['channel']}
                Issue Type (pre-tagged): {ticket.get('main_issue_type', 'unknown')}
                Customer Message: {ticket['latest_message']}

                Acknowledge this ticket and confirm routing to the Classifier.
                """
            ).strip()
        )
        response = llm.invoke([system_prompt, human_message])

        return {
            "messages": [response],
            "next_agent": "classifier",
        }

    next_node = route(state)

    system_prompt = SystemMessage(content=SECOND_PASS_SYSTEM_PROMPT)
    human_message = HumanMessage(
        content=dedent(
            f"""
            Classification received for Ticket {ticket['ticket_id']}:
            - Issue Type: {classification['issue_type']}
            - Urgency: {classification['urgency']}
            - Intent: {classification['intent']}
            - Confidence: {classification['confidence']}

            Routing decision: {next_node.upper()}

            Confirm this routing decision in one sentence.
            """
        ).strip()
    )
    response = llm.invoke([system_prompt, human_message])

    return {
        "messages": [response],
        "next_agent": next_node,
    }
