# UDA-Hub — System Architecture

## Overview

UDA-Hub (Universal Decision Agent Hub) is a LangGraph-powered multi-agent customer support system built for **CultPass** — an experiences and subscription platform where users discover, book, and manage experiential activities via subscription tiers.

UDA-Hub sits as an intelligent layer on top of existing support infrastructure. It reads incoming tickets, reasons about them, routes them to the right agent, retrieves relevant knowledge, and either resolves or escalates — all autonomously.

---

## System Context

```
CultPass Platform  →  Support Tickets  →  UDA-Hub  →  Resolution / Escalation
                                               ↕
                                         Human Agents
```

UDA-Hub connects to two databases:

| Database | Purpose |
|---|---|
| `udahub.db` | Core support system — tickets, messages, knowledge base |
| `cultpass.db` | Product database — users, subscriptions, experiences, reservations |

---

## High-Level Architecture

```
Incoming Ticket (text + metadata)
       ↓
 Memory Hydration Node      ← builds short-term memory + loads long-term memory
       ↓
    Supervisor Agent        ← orchestration brain, no tools
       ↓
    Classifier Agent        ← structured classification, no tools
       ↓
 Memory Hydration Node      ← refreshes memory context after classification
       ↓
    Supervisor Agent
       ↓
   ┌─────┴──────┐
Resolver      Escalation
  Agent          Agent
   ↓               ↓
  END             END
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph |
| LLM | OpenAI GPT-4o-mini |
| Structured output | Pydantic + `with_structured_output()` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (persisted to `data/core/chroma/`) |
| Tool server | FastMCP + FastAPI |
| ORM | SQLAlchemy |
| Short-term memory | LangGraph `MemorySaver` (in-session checkpointing) |
| Long-term memory | `customer_memory` table in `udahub.db` |
| Testing | pytest + RAGAS |

---

## Folder Structure

```
solution/
├── agentic/
│   ├── agents/         ← agent node functions
│   ├── tools/          ← MCP client wrappers
│   ├── design/         ← architecture and responsibility docs
│   ├── logging_config.py
│   └── workflow.py     ← LangGraph graph assembly
├── data/
│   ├── core/           ← udahub.db + ChromaDB vector store
│   ├── external/       ← cultpass.db and seed data
│   └── models/         ← SQLAlchemy models + LangGraph state schema
├── server/
│   ├── main.py         ← FastMCP + FastAPI entry point
│   ├── tools/          ← tool endpoint implementations
│   └── dependencies.py ← shared DB sessions
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.ipynb
├── README.md
├── requirements.txt
└── utils.py
```

---

## Data Flow

### Per-ticket execution flow

```
1. Ticket arrives → initial TicketState assembled
2. START → Memory Hydration
   - Builds short-term memory from thread message history
   - Loads long-term memory by `ticket.external_user_id`
3. Memory → Supervisor (first pass)
   - Acknowledges ticket
   - Sets next_agent = "classifier"
4. Supervisor → Classifier
   - Classifies issue_type, urgency, intent, confidence
   - Writes state["classification"]
5. Classifier → Memory Hydration → Supervisor (second pass)
   - Refreshes memory context before routing decision
   - Routes to escalation when issue type is sensitive
   - Routes to escalation when urgency is high
   - Routes to escalation for phone channel + medium/high urgency
   - Routes to escalation when confidence < 0.6
   - Otherwise routes to resolver
6a. Supervisor → Resolver
   - Gathers context: RAG search + customer lookup + reservation lookup
   - Reasons over all context
   - Executes action tools if confident
   - Writes `tool_usage` for thread-scoped inspection
   - Sets resolved=True → END
   - Sets resolved=False → next_agent = "escalation"
6b. Supervisor → Escalation
   - Determines escalation trigger
   - Produces customer-facing message + internal briefing note
   - Updates ticket status to "escalated"
   - Writes `tool_usage` for thread-scoped inspection
   - Always sets resolved=False → END
```

---

## State Schema

The shared `TicketState` object flows through every node. Each agent reads from and writes only to its own slice — analogous to Redux reducers.

| Field | Type | Written by |
|---|---|---|
| `ticket` | `TicketInput` | Ingestion (before graph runs) |
| `messages` | `list` (append-only) | All agents |
| `tool_usage` | `list[str]` (append-only) | Resolver / Escalation |
| `classification` | `Classification` | Classifier |
| `customer_context` | `CustomerContext` | Resolver |
| `resolution` | `Resolution` | Resolver / Escalation |
| `next_agent` | `str` | Supervisor / Resolver |
| `short_term_memory` | `dict` | Memory Hydration node |
| `long_term_memory` | `dict` | Memory Hydration node + Resolver update path |
| `retrieved_context` | `list[str]` | Resolver (RAG) |
| `error` | `str` | Any agent on failure |

---

## Memory Architecture

### Short-term memory
- Scoped to a single ticket session
- Backed by LangGraph `MemorySaver` checkpointer + `thread_id`
- Materialized by the Memory Hydration node into `short_term_memory.prior_turns`
- Uses a compact rolling window of recent turns to keep prompts bounded

### Long-term memory
- Persists across sessions in `udahub.db` (`customer_memory`)
- Stores: user preferences, past issue types, past resolutions
- Loaded by Memory Hydration using `external_user_id`
- Written by Resolver on successful resolution (`update_long_term_memory`)
- Used by Classifier, Resolver, and Escalation prompts for personalization

---

## Tool Architecture

Tools are exposed as MCP endpoints via a FastMCP + FastAPI server running locally. Agents call tools via MCP client wrappers — they never access databases directly.

```
Agent (agentic/tools/*.py — MCP client)
        ↓  HTTP / MCP protocol
FastMCP Server (server/tools/*.py)
        ↓  SQLAlchemy ORM
udahub.db / cultpass.db / ChromaDB
```

### Tool assignment per agent

| Tool | Supervisor | Classifier | Resolver | Escalation |
|---|---|---|---|---|
| `search_knowledge_base` | | | ✓ | |
| `lookup_customer` | | | ✓ | |
| `lookup_reservation` | | | ✓ | |
| `update_ticket_status` | | | ✓ | ✓ |
| `send_response` | | | ✓ | ✓ |
| `create_internal_note` | | | | ✓ |
| `update_long_term_memory` | | | ✓ | |

---

## Escalation Triggers

| Trigger | Condition |
|---|---|
| Sensitive issue type | `issue_type` in `{legal, abuse, fraud, data_breach}` |
| High urgency | `classification.urgency == "high"` |
| Phone + urgency rule | `ticket.channel == "phone"` and urgency in `{medium, high}` |
| Low confidence | `classification.confidence < 0.6` |
| Resolver cannot act | `resolver.resolved = False` |

---

## Environment Variables

```env
# LLM
VOCAREUM_KEY=

# Database paths
UDAHUB_DB_PATH=data/core/udahub.db
CULTPASS_DB_PATH=data/external/cultpass.db
CHROMA_PATH=data/core/chroma

# FastMCP Server
MCP_SERVER_URL=http://localhost:8000
```
