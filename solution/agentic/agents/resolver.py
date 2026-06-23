import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from textwrap import dedent
from pydantic import BaseModel, Field
from data.models.state import TicketState

from agentic.tools.kb_tools import search_knowledge_base
from agentic.tools.crm_tools import lookup_customer, lookup_reservation, issue_refund
from agentic.tools.ticket_tools import (
    update_ticket_status,
    send_response,
    create_internal_note,
    update_long_term_memory,
)
from agentic.logging_config import (
    log_tool_call,
    log_tool_result,
    log_resolution_attempt,
    log_resolution_success,
    log_resolution_failed,
    log_workflow_end
)


class ResolutionOutput(BaseModel):
    action_taken: str = Field(
        description=(
            "What the Resolver did. Examples: "
            "'answered_from_knowledge_base', 'refund_issued', "
            "'reservation_cancelled', 'account_info_provided', "
            "'cannot_resolve' (use this when escalation is needed)"
        )
    )
    response_message: str = Field(
        description=(
            "The response message to send to the customer. "
            "Should be clear, empathetic, and actionable. "
            "If cannot_resolve, explain that the issue is being escalated."
        )
    )
    tool_calls_made: list[str] = Field(
        default_factory=list,
        description="List of tool names that were called during resolution."
    )
    resolved: bool = Field(
        description=(
            "True if the issue was fully resolved. "
            "False if the Resolver could not confidently act and "
            "the ticket should be escalated."
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Resolver confidence in the resolution. "
            "If below 0.5, set resolved=False and escalate."
        )
    )
    reasoning: str = Field(
        description="Brief internal reasoning for the resolution decision. For audit/debug."
    )



llm = ChatOpenAI(model="gpt-4o-mini",
                temperature=0,
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("VOCAREUM_KEY")
                )
structured_llm = llm.with_structured_output(ResolutionOutput)

SYSTEM_PROMPT = """
You are the Resolver Agent for UDA-Hub, an intelligent customer support system
for CultPass — an experiences and subscription platform.

Your job is to resolve customer support tickets using the context provided.
You have access to:
- Customer profile and subscription details (from CultPass)
- Active reservations and experience details (from CultPass)
- Knowledge base articles and FAQs (from RAG retrieval)
- Prior conversation history (from short-term memory)
- Customer preferences and past resolutions (from long-term memory)

Resolution rules:
- Always check the knowledge base first for FAQ-type questions
- Always look up the customer profile for account/subscription/reservation issues
- Issue refunds only when: explicitly requested AND subscription is active AND reservation exists
- Cancel reservations only when: explicitly requested AND reservation status is 'confirmed'
- If you cannot act with confidence (missing context, ambiguous request, blocked account
  requiring human review), set resolved=False so the ticket escalates
- Never make up information — if you don't have the data, say so and set resolved=False
- Be empathetic, clear, and concise in response_message

Escalate (resolved=False) when:
- Customer is blocked and the block reason is unclear
- Refund amount is disputed or unusually large
- Issue involves multiple conflicting reservations
- Request is outside your tool capabilities
"""

def _gather_context(state: TicketState) -> dict:
    """
    Calls relevant tools based on the classification intent
    to gather context before the LLM reasons.

    Returns a dict of gathered context strings.
    """
    ticket = state["ticket"]
    classification = state["classification"]
    intent = classification["intent"]
    issue_type = classification["issue_type"]
    ticket_id = ticket["ticket_id"]

    context = {
        "kb_results": "",
        "customer_profile": "",
        "reservation_details": "",
        "tools_called": [],
    }

    # Always search the knowledge base
    log_tool_call(ticket_id, "search_knowledge_base", {"query": ticket["latest_message"], "top_k": 3})
    kb_result = search_knowledge_base(
        query=ticket["latest_message"],
        account_id=ticket["account_id"],
        top_k=3,
    )
    log_tool_result(ticket_id, "search_knowledge_base", "success", "Retrieved knowledge base articles")
    context["kb_results"] = kb_result
    context["tools_called"].append("search_knowledge_base")

    # Look up customer for account/subscription/reservation issues
    if issue_type in {"reservation", "subscription", "refund", "account", "billing", "experience"}:
        log_tool_call(ticket_id, "lookup_customer", {"external_user_id": ticket["external_user_id"]})
        customer = lookup_customer(
            external_user_id=ticket["external_user_id"],
            account_id=ticket["account_id"],
        )
        log_tool_result(ticket_id, "lookup_customer", "success", "Retrieved customer profile")
        context["customer_profile"] = customer
        context["tools_called"].append("lookup_customer")

    if issue_type in {"reservation", "refund"} or "reservation" in intent:
        log_tool_call(ticket_id, "lookup_reservation", {"external_user_id": ticket["external_user_id"]})
        reservation = lookup_reservation(
            external_user_id=ticket["external_user_id"],
        )
        log_tool_result(ticket_id, "lookup_reservation", "success", "Retrieved reservation details")
        context["reservation_details"] = reservation
        context["tools_called"].append("lookup_reservation")

    return context



async def run(state: TicketState) -> dict:
    """
    Resolver node function (async).

    1. Gathers context via tools (stubs for now)
    2. Builds a rich prompt with all available context
    3. Invokes structured LLM to produce a resolution decision
    4. Executes any action tools required (refund, status update, etc.)
    5. Returns partial state update

    Returns:
        Partial state update dict with 'resolution', 'customer_context',
        'retrieved_context', and 'messages'.
    """
    ticket = state["ticket"]
    classification = state["classification"]
    short_term = state.get("short_term_memory", {})
    long_term = state.get("long_term_memory", {})

    # ── Step 1: Gather context via tools ──
    gathered = _gather_context(state)

    # ── Step 2: Build prompt ──
    human_message = HumanMessage(
        content=dedent(
            f"""
            Resolve the following support ticket.

            --- TICKET ---
            Ticket ID: {ticket['ticket_id']}
            Channel: {ticket['channel']}
            Customer Message: {ticket['latest_message']}

            --- CLASSIFICATION ---
            Issue Type: {classification['issue_type']}
            Urgency: {classification['urgency']}
            Intent: {classification['intent']}
            Confidence: {classification['confidence']}

            --- CUSTOMER PROFILE (CultPass) ---
            {gathered['customer_profile'] or 'Not retrieved'}

            --- RESERVATION DETAILS (CultPass) ---
            {gathered['reservation_details'] or 'Not applicable'}

            --- KNOWLEDGE BASE RESULTS ---
            {gathered['kb_results'] or 'No relevant articles found'}

            --- CONVERSATION HISTORY (this session) ---
            {short_term.get('prior_turns', 'No prior turns')}

            --- LONG-TERM MEMORY ---
            Known preferences: {long_term.get('preferences', 'None')}
            Past resolutions: {long_term.get('past_resolutions', 'None')}

            Based on all the above, resolve this ticket. If you cannot resolve it confidently,
            set resolved=False so it can be escalated.
            """
        ).strip()
    )

    result: ResolutionOutput = structured_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), human_message]
    )

    # Log resolution attempt
    log_resolution_attempt(ticket["ticket_id"], classification["issue_type"])

    if result.resolved:
        log_tool_call(ticket["ticket_id"], "update_ticket_status", {"status": "resolved"})
        await update_ticket_status(
            ticket_id=ticket["ticket_id"],
            status="resolved",
        )
        log_tool_result(ticket["ticket_id"], "update_ticket_status", "success")
        
        # Send response to customer
        log_tool_call(ticket["ticket_id"], "send_response", {"content_length": len(result.response_message)})
        await send_response(
            ticket_id=ticket["ticket_id"],
            content=result.response_message,
        )
        log_tool_result(ticket["ticket_id"], "send_response", "success")
        
        # Update long-term memory with this resolution
        log_tool_call(ticket["ticket_id"], "update_long_term_memory", {"issue_type": classification["issue_type"]})
        await update_long_term_memory(
            external_user_id=ticket["external_user_id"],
            resolution_summary=result.action_taken,
            issue_type=classification["issue_type"],
        )
        log_tool_result(ticket["ticket_id"], "update_long_term_memory", "success")

        gathered["tools_called"].extend([
            "update_ticket_status",
            "send_response",
            "update_long_term_memory",
        ])
        
        # Log successful resolution
        log_resolution_success(ticket["ticket_id"], result.action_taken, result.confidence)
    else:
        # Log failed resolution
        log_resolution_failed(ticket["ticket_id"], "Resolver could not confidently act")

    resolution = {
        "action_taken": result.action_taken,
        "response_message": result.response_message,
        "tool_calls_made": gathered["tools_called"] + result.tool_calls_made,
        "resolved": result.resolved,
    }

    # Log to message history
    log_message = AIMessage(
        content=(
            f"[Resolver] Ticket {ticket['ticket_id']} — "
            f"action: {result.action_taken} | "
            f"resolved: {result.resolved} | "
            f"confidence: {result.confidence:.2f} | "
            f"reasoning: {result.reasoning}"
        ),
        name="resolver",
    )
    
    # Log workflow end if resolved
    if result.resolved:
        log_workflow_end(ticket["ticket_id"], "resolved")

    return {
        "resolution": resolution,
        "retrieved_context": [gathered["kb_results"]] if gathered["kb_results"] else [],
        "messages": [log_message],
        "next_agent": "end" if result.resolved else "escalation",
    }
