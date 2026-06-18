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
    Supervisor Agent          ← orchestration brain, no tools
         ↓
    Classifier Agent          ← structured classification, no tools
         ↓ (loops back to Supervisor)
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
| LLM | OpenAI GPT-4o |
| Structured output | Pydantic + `with_structured_output()` |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector store | ChromaDB (persisted to `data/core/chroma/`) |
| Tool server | FastMCP + FastAPI |
| ORM | SQLAlchemy |
| Short-term memory | LangGraph `MemorySaver` (in-session checkpointing) |
| Long-term memory | Persistent store (preferences, past resolutions) |
| Testing | pytest + RAGAS |

---

## Folder Structure

```
starter/
├── agentic/
│   ├── agents/         ← agent node functions
│   ├── tools/          ← MCP client wrappers
│   └── workflow.py     ← LangGraph graph assembly
├── data/
│   ├── core/           ← udahub.db + ChromaDB vector store
│   ├── external/       ← cultpass.db
│   └── models/         ← SQLAlchemy models + LangGraph state schema
├── docs/               ← architecture and documentation
├── server/
│   ├── main.py         ← FastMCP + FastAPI entry point
│   ├── routers/        ← tool endpoint implementations
│   └── dependencies.py ← shared DB sessions
├── tests/
│   ├── test_agents/    ← unit tests per agent
│   ├── test_tools/     ← tool unit tests
│   ├── test_workflow.py← end-to-end graph tests
│   └── evaluation/     ← RAGAS RAG quality evaluation
├── .env
├── 01_external_db_setup.ipynb
├── 02_core_db_setup.ipynb
├── 03_agentic_app.ipynb
└── utils.py
```

---

## Data Flow

### Per-ticket execution flow

```
1. Ticket arrives → initial TicketState assembled
2. START → Supervisor (first pass)
   - Acknowledges ticket
   - Sets next_agent = "classifier"
3. Supervisor → Classifier
   - Classifies issue_type, urgency, intent, confidence
   - Writes state["classification"]
4. Classifier → Supervisor (second pass)
   - Reads classification
   - Routes: confidence ≥ 0.6 + non-sensitive → "resolver"
   - Routes: confidence < 0.6 or sensitive type → "escalation"
5a. Supervisor → Resolver
   - Gathers context: RAG search + customer lookup + reservation lookup
   - Reasons over all context
   - Executes action tools if confident
   - Sets resolved=True → END
   - Sets resolved=False → next_agent = "escalation"
5b. Supervisor → Escalation
   - Determines escalation trigger
   - Produces customer-facing message + internal briefing note
   - Updates ticket status to "escalated"
   - Always sets resolved=False → END
```

---

## State Schema

The shared `TicketState` object flows through every node. Each agent reads from and writes only to its own slice — analogous to Redux reducers.

| Field | Type | Written by |
|---|---|---|
| `ticket` | `TicketInput` | Ingestion (before graph runs) |
| `messages` | `list` (append-only) | All agents |
| `classification` | `Classification` | Classifier |
| `customer_context` | `CustomerContext` | Resolver |
| `resolution` | `Resolution` | Resolver / Escalation |
| `next_agent` | `str` | Supervisor / Resolver |
| `short_term_memory` | `dict` | Injected pre-run |
| `long_term_memory` | `dict` | Injected pre-run, updated post-run |
| `retrieved_context` | `list[str]` | Resolver (RAG) |
| `error` | `str` | Any agent on failure |

---

## Memory Architecture

### Short-term memory
- Scoped to a single ticket session
- Managed by LangGraph `MemorySaver` checkpointer
- Stores conversation turns within the same session
- Keyed by `thread_id` (defaults to `ticket_id`)

### Long-term memory
- Persists across sessions
- Stores: user preferences, past issue types, past resolutions
- Read at graph entry, written by Resolver on successful resolution
- Enables better classification and resolution for returning customers

---

## Tool Architecture

Tools are exposed as MCP endpoints via a FastMCP + FastAPI server running locally. Agents call tools via MCP client wrappers — they never access databases directly.

```
Agent (agentic/tools/*.py — MCP client)
        ↓  HTTP / MCP protocol
FastMCP Server (server/routers/*.py)
        ↓  SQLAlchemy ORM
udahub.db / cultpass.db / ChromaDB
```

### Tool assignment per agent

| Tool | Supervisor | Classifier | Resolver | Escalation |
|---|---|---|---|---|
| `search_knowledge_base` | | | ✓ | |
| `lookup_customer` | | | ✓ | |
| `lookup_reservation` | | | ✓ | |
| `issue_refund` | | | ✓ | |
| `update_ticket_status` | | | ✓ | ✓ |
| `send_response` | | | ✓ | ✓ |
| `create_internal_note` | | | ✓ | ✓ |
| `update_long_term_memory` | | | ✓ | |

---

## Escalation Triggers

| Trigger | Condition |
|---|---|
| Sensitive issue type | `issue_type` in `{legal, abuse, fraud, data_breach}` |
| Low confidence | `classification.confidence < 0.6` |
| Resolver cannot act | `resolver.resolved = False` |

---

## Environment Variables

```env
# LLM
OPENAI_API_KEY=

# Database paths
UDAHUB_DB_PATH=data/core/udahub.db
CULTPASS_DB_PATH=data/external/cultpass.db
CHROMA_PATH=data/core/chroma

# FastMCP Server
MCP_SERVER_URL=http://localhost:8000
```
