 AI Framework: Intelligent Retrieval and Multi-Tool Task Orchestration

A modular AI framework that lets users interact with their documents, email, calendar, maps, and structured data through plain natural language. It combines a user-specific RAG pipeline, intelligent request routing, native tools, MCP-based external tool servers, and multi-step orchestration in a single extensible architecture.

> Built during an AI Engineering Internship at Jio Platforms Ltd. (19 June 2026 – 18 September 2026) by Aashutosh Srivastava, under the mentorship of Mr. Shashank Bhushan.

---

 Table of Contents

- [Features](features)
- [Architecture](architecture)
- [Tech Stack](tech-stack)
- [Project Structure](project-structure)
- [Getting Started](getting-started)
- [Usage and Example Queries](usage-and-example-queries)
- [How It Works](how-it-works)
- [Challenges and Solutions](challenges-and-solutions)
- [Testing](testing)
- [Future Scope](future-scope)
- [Acknowledgements](acknowledgements)

---

 Features

- Intelligent request routing: classifies each request as direct LLM processing, retrieval-based, or workflow-based, and builds a multi-step execution plan for complex requests.
- User-specific RAG: per-user document collections with hybrid retrieval (FAISS vector search plus BM25 keyword search), query refinement, and Cohere reranking.
- Multi-tool orchestration: splits a complex request into dependent steps and passes the output of one step as input to the next (for example, geocode a place and then route using the exact coordinates).
- Google Workspace integration (MCP): search and read Gmail, send or draft emails, and list, get, create, and update Google Calendar events, with OAuth-based authentication.
- Maps integration (MCP, OSM-based): geocoding, reverse geocoding, routing, route matrix, route optimization, nearby place search, and place details.
- Native CSV tool: ask questions about CSV files in natural language (filter, search, aggregate, extract) without any external server.
- Authenticated user context: the signed-in user's email is carried through retrieval and Workspace operations.
- Modular and extensible: UI, routing, retrieval, orchestration, native tools, and external tool servers are cleanly separated, so new tools can be added without reworking the core.
- Error handling: input validation, captured MCP and API failures, separate handling of auth errors, and validation of intermediate results before they flow to later steps.

---

 Architecture

```
                    ┌────────────────────────┐
                    │  Streamlit UI Layer    │
                    │  (auth + user context) │
                    └───────────┬────────────┘
                                │ query + user email
                    ┌───────────▼────────────┐
                    │   AI Framework Core    │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │     Request Router     │
                    │ (direct / RAG / crew)  │
                    └─┬──────┬───────┬─────┬─┘
                      │      │       │     │
        ┌─────────────▼┐  ┌──▼─────┐ │  ┌──▼───────────────┐
        │  Retrieval   │  │ Native │ │  │  Orchestration   │
        │  Layer (RAG) │  │  CSV   │ │  │ (CrewAI agents)  │
        └──────────────┘  │  Tool  │ │  └──┬────────────┬──┘
                          └────────┘ │     │ MCP        │ MCP
                                     │  ┌──▼─────────┐ ┌▼────────────┐
                                     │  │ Google     │ │ OSM Maps    │
                                     │  │ Workspace  │ │ MCP Server  │
                                     │  │ MCP Server │ │             │
                                     │  └────────────┘ └─────────────┘
                                     └─ Direct LLM processing
```

 Core components

| Component | Responsibility |
|---|---|
| User Interface (Streamlit) | Accepts natural-language requests, handles authentication, passes the user's email to the framework, and displays results. |
| Framework Core | Central request-processing entry point that coordinates routing, retrieval, orchestration, and tool execution. |
| Request Router | Decides the execution path and can generate a multi-step execution plan. |
| Retrieval Layer | User-specific hybrid RAG pipeline with reranking. |
| Orchestration Layer | Runs multi-step workflows and propagates intermediate results between steps. |
| Agent Workflow Module | CrewAI agents created via a central agent factory, with task definitions and a crew manager. |
| Native CSV Tool | Built-in structured-data tool, no external server required. |
| Google Workspace MCP Server | Separately running server exposing Gmail and Calendar tools. |
| OSM Maps MCP Server | Separately running server exposing geocoding, routing, and place tools. |

---

 Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| API layer | FastAPI |
| UI | Streamlit |
| Vector search | FAISS |
| Document indexing | LlamaIndex |
| Keyword retrieval | BM25 |
| Reranking | Cohere Rerank |
| Agent workflows | CrewAI |
| Tool protocol | Model Context Protocol (MCP) |
| Maps and routing | OpenStreetMap (OSM) server, OpenRouteService API |
| LLMs | Groq and Gemini |
| Tooling | Git, Python virtual environments |

---

 Project Structure

```
ai_framework/
├── main.py                   Application entry point
├── .env                      API keys and environment config (not committed)
├── .streamlit/
│   └── secrets.toml          Streamlit auth/secrets (not committed)
├── config/
│   └── settings.py           Centralized configuration loader
├── framework/
│   └── framework.py          Core request-processing logic
├── crew/
│   ├── agent_factory.py      Centralized agent creation
│   ├── task_factory.py       Task definitions for workflows
│   ├── crew_manager.py       Crew execution and orchestration
│   └── tools/
│       ├── calendar_tools.py
│       ├── csv_tools.py      Native CSV tool
│       ├── gmail_tools.py
│       ├── maps_tools.py
│       └── time_tools.py
└── llm/                      LLM client setup
```

> The Google Workspace MCP server and the OSM Maps MCP server are developed and run separately from this repository, and the framework connects to them over a network-based MCP connection.

---

 Getting Started

 Prerequisites

- Python 3.10+ (developed on Python 3.13 / macOS)
- A stable internet connection
- 8 GB RAM recommended
- API keys for the services you plan to use (Groq, Gemini, Cohere, OpenRouteService)
- A Google Cloud project with OAuth credentials for Gmail and Calendar access
- The Google Workspace MCP server and OSM Maps MCP server running locally or remotely

 Installation

```bash
 1. Clone the repository
git clone https://github.com/aasrivastava27/ai-framework.git
cd ai-framework

 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate         macOS/Linux
 venv\Scripts\activate          Windows

 3. Install dependencies
pip install -r requirements.txt
```

 Configuration

Create a `.env` file in the project root. The variable names below are examples, so match them to what `config/settings.py` expects:

```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
COHERE_API_KEY=your_cohere_key
OPENROUTESERVICE_API_KEY=your_ors_key

 MCP server endpoints
WORKSPACE_MCP_URL=http://localhost:<port>/mcp
MAPS_MCP_URL=http://localhost:<port>/mcp
```

Add your Streamlit authentication settings to `.streamlit/secrets.toml`.

> Never commit `.env` or `secrets.toml`. Make sure both are listed in `.gitignore`.

 Start the MCP servers

Start the Google Workspace MCP server and the OSM Maps MCP server before launching the framework. Refer to each server's own documentation for setup and OAuth configuration.

 Run the application

```bash
python main.py
```

The Streamlit app will be available at `http://localhost:8501`.

---

 Usage and Example Queries

1. Sign in through the Streamlit interface.
2. Optionally upload `.txt` documents or `.csv` files using the upload widget.
3. Type a natural-language request in the chat box.

| Capability | Example request |
|---|---|
| RAG | "Find the information about the AI Framework in my engineering documents." |
| Gmail | "What emails have I received recently regarding my AI project?" |
| Calendar | "Create a calendar event for an AI project meeting tomorrow from 3 PM to 4 PM." |
| CSV | "Analyze the CSV file and find the department with the highest average salary." |
| Maps (chained) | "Find a café near Gateway of India and calculate the route from that café to Bandra Fort." |
| Gmail → Calendar | "Find the latest email from my project team and check my calendar for any related meeting tomorrow." |
| CSV → Gmail | "Analyze test1CSV.csv, find the department with the highest average salary, then send an email to my manager telling them the result." |
| Gmail → Calendar (computed) | "Find the latest mail from the Admissions Officer and create a calendar event three months after the date in the mail." |

 Example: multi-step CSV → Email workflow

```
Request: "Analyze test1CSV.csv, find the department with the highest average
          salary, then send an email to my manager telling them the result."

Execution plan:
  1. crew → data    : Analyze the CSV and calculate average salary per department
  2. llm            : Identify the department with the highest average salary
  3. crew → gmail   : Send an email informing the manager of the result

Result: Engineering (average salary 120000.0), email sent successfully.
```

---

 How It Works

 Request lifecycle

```
User Request → Authentication & User Context → Framework Request Processing
→ Request Routing → Capability Selection
→ Retrieval / Native Tool / External MCP Tool Execution
→ Intermediate Result Processing → Multi-Step Orchestration (if needed)
→ Result Consolidation → Final Response
```

 RAG pipeline

1. Documents are stored in user-specific directories, keyed by the authenticated Google account.
2. Documents are split into overlapping chunks and indexed.
3. Vector retrieval (FAISS) finds semantically similar chunks.
4. Keyword retrieval (BM25) runs in parallel to catch exact term matches.
5. Query refinement and multiple query formulations improve coverage.
6. Candidates from both strategies are merged and reranked (Cohere).
7. The top context is passed to the LLM to produce a grounded answer.

 Multi-step orchestration

The orchestrator determines the order of operations, calls each tool through its interface, captures structured intermediate results, validates them, and passes them to dependent steps. For example, exact latitude and longitude returned by a nearby-place search are passed directly to the routing tool instead of re-resolving an ambiguous place name.

---

 Challenges and Solutions

| Challenge | Solution |
|---|---|
| Authentication and user context | Identity is obtained at the app layer and passed consistently through retrieval and Workspace calls. |
| Retrieval accuracy | Hybrid semantic and keyword retrieval, query refinement, multi-query variations, and reranking. |
| Multi-step orchestration | Explicit dependency management with stored intermediate results between steps. |
| Tool input/output propagation | Extract the relevant fields from prior results, convert them to the next tool's input format, and validate before propagating. |
| MCP integration | Clear interfaces for connection management, tool discovery, and structured tool invocation. |
| TPM and TPD rate limits | Incremental testing, fine-tuned request handling, and avoiding unnecessary large workflows. |
| Maintainability | Core logic separated from native tools and from external MCP servers. |

---

 Testing

Testing was done incrementally, from individual modules up to full workflows:

- Routing tests to confirm requests reach the correct pipeline
- RAG tests with long text documents, varied phrasing, and grounding checks against sources
- Native CSV tests for filtering, searching, aggregation, and extraction against expected values
- Gmail and Calendar tests for search, retrieval, creation, updates, and auth behavior
- Maps tests for geocoding, reverse geocoding, nearby search, place details, and routing
- Multi-step workflow tests using real intermediate outputs, plus failure scenarios
- Integration tests across the UI, core, router, retrieval, native tools, and MCP servers

---

 Future Scope

- More file formats for the RAG pipeline beyond the currently supported ones
- Separate per-user document storage for better isolation and scalability
- Enhanced user context management across workflows
- Improved error recovery with retries, fallbacks, and partial-result handling
- Reduced rate-limit constraints through caching, efficient token usage, and better request management

---

 Acknowledgements

- Jio Platforms Ltd. for the internship opportunity
- Mr. Shashank Bhushan for mentorship and guidance throughout the project

---

 Author

Aashutosh Srivastava
GitHub: [@aasrivastava27](https://github.com/aasrivastava27)
