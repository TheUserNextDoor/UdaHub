from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from textwrap import dedent
from data.models.state import TicketState
import os

from agentic.tools.ticket_tools import (
    update_ticket_status,
    send_response,
    create_internal_note,
)

class EscalationOutput(BaseModel):
    escalation_reason: str = Field(
        description=(
            "Why this ticket is being escalated. Examples: "
            "'low_classifier_confidence', 'sensitive_issue_type', "
            "'resolver_cannot_act', 'blocked_account_review', "
            "'disputed_refund', 'fraud_suspected'"
        )
    )
    customer_message: str = Field(
        description=(
            "Empathetic holding message to send to the customer. "
            "Should acknowledge their issue, assure them a human agent "
            "will follow up, and give a realistic expectation. "
            "Do not reveal internal reasoning or system details."
        )
    )
    internal_note: str = Field(
        description=(
            "Structured briefing note for the human agent taking over. "
            "Should include: issue summary, what was attempted, "
            "relevant customer context, and suggested next steps. "
            "Be factual and concise."
        )
    )
    urgency_flag: str = Field(
        description=(
            "Urgency flag for the human agent queue. "
            "Must be one of: 'low', 'medium', 'high', 'critical'. "
            "Use 'critical' only for fraud, data_breach, or abuse."
        )
    )
    reasoning: str = Field(
        description="Brief internal reasoning for audit and debugging."
    )

llm = ChatOpenAI(model="gpt-4o-mini",
                temperature=0,
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("VOCAREUM_KEY")
                )
structured_llm = llm.with_structured_output(EscalationOutput)

SYSTEM_PROMPT = """
You are the Escalation Agent for UDA-Hub, an intelligent customer support system
for CultPass — an experiences and subscription platform.

Your job is to handle tickets that cannot be automatically resolved. This happens when:
- The issue is sensitive (legal, fraud, abuse, data_breach)
- The Classifier had low confidence in its classification
- The Resolver attempted resolution but could not act confidently

Your two outputs serve different audiences:
1. customer_message — empathetic, reassuring, professional. The customer reads this.
   Never reveal internal system details, confidence scores, or agent names.
2. internal_note — factual, structured, actionable. The human agent reads this.
   Include everything they need to pick up the ticket without re-reading the history.

Urgency escalation rules:
- 'critical': fraud, data_breach, abuse, legal threat
- 'high': blocked account, disputed charge, premium tier issue
- 'medium': reservation problem, subscription billing, unresolved complaint
- 'low': general enquiry that needs human touch, unclear request
"""

def _determine_escalation_trigger(state: TicketState) -> str:
    """
    Derives the escalation trigger from state for prompt context.
    """
    classification = state.get("classification")
    resolution = state.get("resolution")

    sensitive_types = {"legal", "abuse", "fraud", "data_breach"}

    if classification and classification["issue_type"] in sensitive_types:
        return f"Sensitive issue type detected: '{classification['issue_type']}'"

    if classification and classification["confidence"] < 0.6:
        return f"Low classification confidence: {classification['confidence']:.2f}"

    if resolution and not resolution["resolved"]:
        return f"Resolver could not act: '{resolution['action_taken']}'"

    return "Escalation triggered by Supervisor routing decision"


async def run(state: TicketState) -> dict:
    """
    Escalation node function (async).

    1. Determines why escalation was triggered
    2. Assembles all available context from state
    3. Invokes structured LLM to produce customer message + internal note
    4. Executes ticket tools: status update, send response, create note
    5. Returns partial state update

    Returns:
        Partial state update dict with 'resolution' and 'messages'.
    """
    ticket = state["ticket"]
    classification = state.get("classification")
    resolution = state.get("resolution")
    customer_context = state.get("customer_context")
    short_term = state.get("short_term_memory", {})
    long_term = state.get("long_term_memory", {})

    escalation_trigger = _determine_escalation_trigger(state)

    # ── Build prompt ──
    human_message = HumanMessage(
        content=dedent(
            f"""
            Handle the escalation of the following support ticket.

            --- TICKET ---
            Ticket ID: {ticket['ticket_id']}
            Channel: {ticket['channel']}
            Customer Message: {ticket['latest_message']}

            --- ESCALATION TRIGGER ---
            {escalation_trigger}

            --- CLASSIFICATION (if available) ---
            Issue Type: {classification['issue_type'] if classification else 'Not classified'}
            Urgency: {classification['urgency'] if classification else 'Unknown'}
            Intent: {classification['intent'] if classification else 'Unknown'}
            Confidence: {classification['confidence'] if classification else 'N/A'}

            --- RESOLVER ATTEMPT (if applicable) ---
            Action Attempted: {resolution['action_taken'] if resolution else 'No resolution attempted'}
            Resolver Response: {resolution['response_message'] if resolution else 'N/A'}

            --- CUSTOMER CONTEXT (if retrieved) ---
            {customer_context if customer_context else 'Not retrieved'}

            --- CONVERSATION HISTORY (this session) ---
            {short_term.get('prior_turns', 'No prior turns')}

            --- LONG-TERM MEMORY ---
            Known preferences: {long_term.get('preferences', 'None')}
            Past resolutions: {long_term.get('past_resolutions', 'None')}

            Produce:
            1. An empathetic customer-facing message
            2. A structured internal briefing note for the human agent
            3. An urgency flag for the queue
            """
        ).strip()
    )

    result: EscalationOutput = structured_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), human_message]
    )

    await update_ticket_status(
        ticket_id=ticket["ticket_id"],
        status="escalated",
        urgency=result.urgency_flag,
    )

    await send_response(
        ticket_id=ticket["ticket_id"],
        content=result.customer_message,
        role="agent",
    )

    await create_internal_note(
        ticket_id=ticket["ticket_id"],
        content=result.internal_note,
        role="system",
    )

    resolution_update = {
        "action_taken": f"escalated — {result.escalation_reason}",
        "response_message": result.customer_message,
        "tool_calls_made": [
            "update_ticket_status",
            "send_response",
            "create_internal_note",
        ],
        "resolved": False, 
    }

    log_message = AIMessage(
        content=(
            f"[Escalation] Ticket {ticket['ticket_id']} escalated — "
            f"reason: {result.escalation_reason} | "
            f"urgency: {result.urgency_flag} | "
            f"reasoning: {result.reasoning}"
        ),
        name="escalation",
    )

    return {
        "resolution": resolution_update,
        "next_agent": "end",
        "messages": [log_message],
    }
