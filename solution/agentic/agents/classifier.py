from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from textwrap import dedent
from data.models.state import TicketState
import os


class ClassificationOutput(BaseModel):
    issue_type: str = Field(
        description=(
            "Category of the support issue. Must be one of: "
            "'reservation', 'subscription', 'refund', 'account', "
            "'experience', 'billing', 'legal', 'abuse', 'fraud', "
            "'data_breach', 'general'"
        )
    )
    urgency: str = Field(
        description=(
            "Urgency level of the ticket. Must be one of: "
            "'low', 'medium', 'high'"
        )
    )
    intent: str = Field(
        description=(
            "The specific action or outcome the customer wants. "
            "Examples: 'cancel_reservation', 'upgrade_plan', "
            "'request_refund', 'unlock_account', 'get_info', "
            "'report_issue', 'dispute_charge'"
        )
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Confidence score between 0.0 and 1.0 indicating how certain "
            "the classifier is about this classification. "
            "Use lower scores when the ticket is ambiguous or lacks context."
        )
    )
    reasoning: str = Field(
        description=(
            "Brief one-sentence explanation of why this classification "
            "was chosen. Used for debugging and audit trails."
        )
    )


llm = ChatOpenAI(model="gpt-4o-mini",
                temperature=0,
                base_url="https://openai.vocareum.com/v1",
                api_key=os.getenv("VOCAREUM_KEY")
                )

structured_llm = llm.with_structured_output(ClassificationOutput)


SYSTEM_PROMPT = """
You are the Classifier Agent for UDA-Hub, an intelligent customer support system
for CultPass — an experiences and subscription platform where users discover,
book, and manage experiential activities (concerts, workshops, tours, dining, etc.)
via subscription tiers.

Your job is to analyse an incoming support ticket and classify it accurately.

CultPass context:
- Users have subscription tiers (e.g. basic, premium) with monthly booking quotas
- Users can reserve experiences (events, activities)
- Common issues: reservation problems, subscription billing, refund requests,
  account access, experience quality complaints

Classification rules:
- Choose issue_type from the allowed values only
- Set urgency to 'high' if: payment dispute, account blocked, data concern, abusive language
- Set urgency to 'medium' if: reservation issue, subscription problem, billing query
- Set urgency to 'low' if: general information request, minor complaint
- Set confidence below 0.6 if: the message is vague, too short, or contradictory
- Flag as 'legal', 'abuse', 'fraud', or 'data_breach' only when explicitly indicated
"""


async def run(state: TicketState) -> dict:
    """
    Classifier node function (async).

    Reads the ticket from state, runs structured LLM classification,
    and returns the classification to be merged into state.

    Returns:
        Partial state update dict with 'classification' and 'messages'.
    """
    ticket = state["ticket"]

    prior_messages = state.get("short_term_memory", {}).get("prior_turns", "")
    history_block = (
        f"\nPrior conversation history:\n{prior_messages}"
        if prior_messages
        else ""
    )

    long_term = state.get("long_term_memory", {})
    known_preferences = long_term.get("preferences", "")
    past_issue_types = long_term.get("past_issue_types", [])

    memory_block = ""
    if known_preferences or past_issue_types:
        memory_block = dedent(
            f"""
            Known customer preferences: {known_preferences or 'none'}
            Past issue types raised: {', '.join(past_issue_types) if past_issue_types else 'none'}
            """
        ).strip()

    human_message = HumanMessage(
        content=dedent(
            f"""
            Classify the following support ticket.

            Ticket ID: {ticket['ticket_id']}
            Channel: {ticket['channel']}
            Pre-tagged issue type (may be empty or incorrect): {ticket.get('main_issue_type', 'unknown')}
            Pre-tagged tags: {ticket.get('tags', 'none')}
            {history_block}
            {memory_block}
            Customer message:
            \"\"\"{ticket['latest_message']}\"\"\"

            Return a structured classification.
            """
        ).strip()
    )


    result: ClassificationOutput = structured_llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), human_message]
    )

    classification = {
        "issue_type": result.issue_type,
        "urgency": result.urgency,
        "intent": result.intent,
        "confidence": result.confidence,
    }

    confirmation = HumanMessage(
        content=(
            f"[Classifier] Ticket {ticket['ticket_id']} classified as "
            f"'{result.issue_type}' | urgency: {result.urgency} | "
            f"intent: {result.intent} | confidence: {result.confidence:.2f} | "
            f"reasoning: {result.reasoning}"
        ),
        name="classifier",
    )

    return {
        "classification": classification,
        "messages": [confirmation],
    }
