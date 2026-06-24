# UDA-Hub — Agent Responsibilities

Quick reference for each agent's role, inputs, outputs, tools, and escalation behaviour.

---

## Supervisor Agent

**File:** `agentic/agents/supervisor.py`

**Role:** Orchestration brain. Entry point for every ticket. Routes between agents based on state. Never resolves tickets itself.

**Tools:** None

**Reads from state:**
- `ticket` — ticket content and metadata
- `classification` — output from Classifier (second pass)
- `resolution` — checks if already resolved
- `next_agent` — current routing intention

**Writes to state:**
- `next_agent` — routing decision
- `messages` — acknowledgement and routing confirmation logs

**Routing logic:**

| Condition | Routes to |
|---|---|
| No classification yet | `classifier` |
| Already resolved | `END` |
| Sensitive issue type (legal, abuse, fraud, data_breach) | `escalation` |
| `classification.confidence < 0.6` | `escalation` |
| All other cases | `resolver` |

**Passes:** 2 per ticket
1. First pass — no classification, acknowledges ticket, routes to Classifier
2. Second pass — classification received, evaluates and routes to Resolver or Escalation

---

## Classifier Agent

**File:** `agentic/agents/classifier.py`

**Role:** Labels the ticket with structured classification. Pure LLM reasoning — no tools, no DB access.

**Tools:** None

**Reads from state:**
- `ticket` — ticket content, channel, pre-tagged metadata
- `short_term_memory.prior_turns` — conversation history
- `long_term_memory.preferences` — known customer preferences
- `long_term_memory.past_issue_types` — prior issue history

**Writes to state:**
- `classification` — structured output: `issue_type`, `urgency`, `intent`, `confidence`
- `messages` — classification log with reasoning

**Output schema:**

| Field | Type | Values |
|---|---|---|
| `issue_type` | `str` | reservation, subscription, refund, account, experience, billing, legal, abuse, fraud, data_breach, general |
| `urgency` | `str` | low, medium, high |
| `intent` | `str` | cancel_reservation, upgrade_plan, request_refund, unlock_account, get_info, report_issue, dispute_charge, … |
| `confidence` | `float` | 0.0 – 1.0 |

**Confidence rules:**
- Below `0.6` → Supervisor will escalate
- Use lower scores for: vague messages, too short, contradictory intent
- Use higher scores for: clear, specific, well-structured tickets

**Always routes back to:** Supervisor (fixed edge)

---

## Resolver Agent

**File:** `agentic/agents/resolver.py`

**Role:** The action-taking agent. Retrieves context, reasons over it, and either resolves the ticket or admits it cannot and hands off to Escalation.

**Tools:**

| Tool | Source DB | When called |
|---|---|---|
| `search_knowledge_base` | ChromaDB (udahub) | Always — RAG retrieval |
| `lookup_customer` | cultpass.db | account, subscription, refund, reservation, billing issues |
| `lookup_reservation` | cultpass.db | reservation or refund intents |
| `issue_refund` | cultpass.db | explicit refund request + active subscription + reservation exists |
| `update_ticket_status` | udahub.db | on resolved=True only |
| `send_response` | udahub.db | on resolved=True only |
| `update_long_term_memory` | memory store | on resolved=True only |

**Reads from state:**
- `ticket` — ticket and customer identifiers
- `classification` — issue type and intent drives tool selection
- `short_term_memory` — prior conversation turns
- `long_term_memory` — preferences and past resolutions

**Writes to state:**
- `resolution` — action taken, response message, tools called, resolved flag
- `customer_context` — CultPass profile fetched during execution
- `retrieved_context` — RAG chunks returned from knowledge base
- `next_agent` — "end" if resolved, "escalation" if not
- `messages` — resolution log with reasoning

**Resolution rules:**
- Issue refunds only when: explicitly requested AND subscription active AND reservation confirmed
- Cancel reservations only when: explicitly requested AND status is confirmed
- Never fabricate information — set `resolved=False` if context is missing
- Escalate if: customer is blocked, refund is disputed, issue is outside tool capabilities

**Output schema:**

| Field | Type | Description |
|---|---|---|
| `action_taken` | `str` | What was done (e.g. `refund_issued`, `answered_from_knowledge_base`) |
| `response_message` | `str` | Message sent to customer |
| `tool_calls_made` | `list[str]` | Tools executed |
| `resolved` | `bool` | True = done, False = escalate |

---

## Escalation Agent

**File:** `agentic/agents/escalation.py`

**Role:** Terminal handler for unresolvable tickets. Produces two outputs: a customer-facing holding message and an internal briefing note for the human agent taking over.

**Tools:**

| Tool | Source DB | Purpose |
|---|---|---|
| `update_ticket_status` | udahub.db | Sets status to "escalated" with urgency flag |
| `send_response` | udahub.db | Sends holding message to customer |
| `create_internal_note` | udahub.db | Briefing note for human agent |

**Reads from state:**
- `ticket` — ticket content
- `classification` — if available, used to shape the briefing note
- `resolution` — Resolver's failed attempt (if applicable)
- `customer_context` — if Resolver retrieved it
- `short_term_memory` — conversation history
- `long_term_memory` — preferences and past resolutions

**Writes to state:**
- `resolution` — action: "escalated — {reason}", resolved: False (always)
- `next_agent` — always "end"
- `messages` — escalation log with reason and urgency

**Escalation triggers:**

| Trigger | Source |
|---|---|
| Sensitive issue type | Supervisor detects before Classifier runs |
| Low classifier confidence | Supervisor reads `classification.confidence < 0.6` |
| Resolver cannot act | Resolver sets `resolved=False` |

**Urgency flags:**

| Flag | When |
|---|---|
| `critical` | fraud, data_breach, abuse, legal threat |
| `high` | blocked account, disputed charge, premium tier issue |
| `medium` | reservation problem, subscription billing, unresolved complaint |
| `low` | general enquiry needing human touch |

**Output schema:**

| Field | Type | Description |
|---|---|---|
| `escalation_reason` | `str` | Why escalated (internal) |
| `customer_message` | `str` | Empathetic holding message — never reveals internal details |
| `internal_note` | `str` | Structured briefing for human agent |
| `urgency_flag` | `str` | low / medium / high / critical |

**Always terminal** — `resolved` is hardcoded to `False`. Only a human agent can close an escalated ticket.

---

## Agent Interaction Summary

```
START
  ↓
Supervisor (pass 1) → acknowledges ticket → next_agent = "classifier"
  ↓
Classifier → classifies ticket → loops back to Supervisor
  ↓
Supervisor (pass 2) → evaluates classification → routes
  ↓                                               ↓
Resolver                                     Escalation
  ↓ resolved=True → END                          ↓
  ↓ resolved=False → Escalation             always END
```
