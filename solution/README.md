# UdaHub: Multi-Agent Support System with LangGraph

A production-ready intelligent customer support system for CultPass using LangGraph, LangChain, and FastMCP. Demonstrates multi-agent orchestration, knowledge-based retrieval, persistent customer memory, and context-aware escalation.

## Getting Started

This project implements an agentic AI system for handling customer support tickets across multiple channels with KB-informed responses, personalized context, and intelligent routing.

### Dependencies

```
fastmcp>=2.10.6
httpx>=0.28.1
ipykernel>=6.30.0
langchain>=0.3.27
langchain-chroma>=0.4.24
langchain-core>=0.3.72
langchain-mcp-adapters>=0.1.9
langchain-openai>=0.3.28
langgraph-supervisor>=0.0.28
langgraph>=0.5.4
python-dotenv>=1.1.1
sqlalchemy>=2.0.41
```

### Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment:**
   ```bash
   cp .env.example .env
   # Add your VOCAREUM_KEY to .env
   export VOCAREUM_KEY="your_key_here"
   ```

3. **Initialize databases:**
   - Run `01_external_db_setup.ipynb` to load external CultPass data (users, articles, experiences)
   - Run `02_core_db_setup.ipynb` to create UdaHub schema and vector store
   - This creates: `data/core/udahub.db`, `data/core/cultpass.db`, Chroma vector index

4. **Start the MCP server** (in separate terminal):
   ```bash
   python -m server.main
   ```
   The server exposes tools for KB search, CRM queries, and ticket operations.

## Running the Application

Execute `03_agentic_app.ipynb`:
- **Cell 1-6**: Setup, load environment, import the orchestrator
- **Cell 7**: Main orchestrator execution (`await chat_interface(orchestrator, "1")`)
- **Cell 8**: View conversation history and state traces
- **Cell 10**: Routing tests with sample tickets (KB hit/miss, escalation scenarios)
- **Cell 12**: Direct tool usage tests (lookup_customer, lookup_reservation, issue_refund)
- **Cell 14**: Persistent memory & personalization demo (first visit → resolution → return visit)
- **Cell 16**: Structured log inspection (searchable JSON log demo)

## Project Architecture

### Multi-Agent Graph
- **Supervisor**: Routes tickets based on urgency, channel, and issue type
- **Classifier**: Analyzes tickets; outputs issue_type, urgency, intent, confidence
- **Resolver**: Uses KB + CRM to resolve tickets; enforces KB grounding; escalates on miss
- **Escalation**: Terminal node; creates escalation notes with full context

### State Management
- Ticket metadata and messages
- Classification (issue_type, urgency, confidence)
- Short-term memory (conversation history)
- Long-term memory (customer preferences, past resolutions)
- Retrieved context from KB

### Persistent Storage
- **udahub.db**: Tickets, messages, customer memory, KB articles
- **cultpass.db**: User profiles, reservations, billing
- **Chroma**: Vector embeddings for semantic KB search

## Key Features

✅ **Knowledge-Based Responses**: All resolved answers grounded in KB articles
✅ **Persistent Customer Memory**: Stores resolutions, preferences, issue history
✅ **Async Architecture**: Full async propagation through agents and tools
✅ **Intelligent Routing**: Supervisor routes based on urgency, sensitivity, channel
✅ **Context-Aware Support**: Uses customer history for personalized responses
✅ **Tool Abstraction**: FastMCP client wrappers for clean agent tool calls
✅ **Comprehensive Logging**: Structured logs for tool calls, routing, escalations

## Built With

* [LangGraph](https://python.langchain.com/docs/langgraph/) - Multi-agent orchestration
* [LangChain](https://python.langchain.com/) - LLM framework
* [FastMCP](https://github.com/jlowin/fastmcp) - Tool server for MCP protocol
* [SQLAlchemy](https://www.sqlalchemy.org/) - ORM for persistent storage
* [Chroma](https://www.trychroma.com/) - Vector database for KB retrieval
* [OpenAI API](https://openai.com/api/) - LLM (via Vocareum endpoint)
* [Pydantic](https://docs.pydantic.dev/) - Structured outputs and validation

## Project Structure

```
solution/
├── 01_external_db_setup.ipynb      # Load CultPass data
├── 02_core_db_setup.ipynb          # Initialize UdaHub schema
├── 03_agentic_app.ipynb            # Main orchestrator & demos
├── utils.py                         # Database utilities
├── data/
│   ├── core/                       # UdaHub & CultPass DBs, Chroma
│   ├── external/                   # Sample JSON datasets
│   └── models/
│       ├── udahub.py               # SQLAlchemy models (Ticket, CustomerMemory, etc.)
│       ├── cultpass.py             # CultPass schemas
│       └── state.py                # LangGraph state schema
├── agentic/
│   ├── workflow.py                 # Graph assembly & compilation
│   ├── agents/
│   │   ├── supervisor.py           # Routing decisions
│   │   ├── classifier.py           # Issue classification
│   │   ├── resolver.py             # KB-based resolution
│   │   └── escalation.py          # Terminal escalation
│   ├── tools/
│   │   ├── kb_tools.py             # KB search client
│   │   ├── crm_tools.py            # CultPass lookup client
│   │   └── ticket_tools.py         # Ticket & memory operations client
│   ├── logging_config.py           # Structured workflow logging
│   └── design/
│       ├── architecture.md         # System architecture documentation
│       ├── agent_responsibilities.md  # Agent responsibilities reference
│       └── README.md               # Folder note
├── server/
│   └── tools/                      # FastMCP tool implementations
└── README.md                       # This file
```

## License

[License](../../LICENSE.md)
