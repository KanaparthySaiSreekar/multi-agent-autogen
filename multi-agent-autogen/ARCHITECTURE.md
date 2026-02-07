# Backend Architecture

## High-Level Overview

```
User (Browser)
    |
    |  HTTP / SSE
    v
FastAPI Server  (src/main.py)
    |
    |--- /api/documents/*   (src/routes.py)        -- REST API for document CRUD + background analysis
    |--- /api/chat/*        (src/chat_routes.py)    -- Chat SSE endpoint + conversation management
    |--- /  (static files)                          -- Serves built React frontend
    |
    v
Orchestrator  (src/orchestrator.py)
    |
    |--- Intent Classifier  (LLM call)             -- Decides: conversation or analysis?
    |       |
    |       |-- "conversation" --> Single LLM call (simple_chat_streaming)
    |       |-- "analysis"     --> Multi-agent pipeline (analyze_document_streaming)
    |                                   |
    |                                   v
    |                           AG2 GroupChat (round_robin)
    |                             1. SummaryAgent
    |                             2. ActionAgent
    |                             3. RiskAgent
    |
    v
SQLite Database  (db/documents.db)
```

## Is There a "Smart" Main Agent?

**No.** There is no intelligent orchestrator agent making dynamic decisions about which sub-agents to call. Here's exactly what happens:

1. **Intent classification** uses a one-shot LLM call to decide between two hardcoded paths: `"conversation"` or `"analysis"`. It does NOT dynamically pick agents.

2. **If analysis:** ALL three agents (Summary, Action, Risk) ALWAYS run, in a FIXED order (`round_robin`). There is no logic to skip an agent, run agents conditionally, or choose different agents based on the document type.

3. **If conversation:** A single LLM call responds directly. No agents are involved.

The agent pipeline is a **static, sequential chain** -- not a dynamic graph.

---

## File-by-File Breakdown

### `src/main.py` -- Application Entry Point

- Creates the FastAPI app with CORS middleware
- Runs `init_db()` on startup (creates tables if they don't exist)
- Mounts three things:
  - `router` from `src/routes.py` at `/api`
  - `chat_router` from `src/chat_routes.py` at `/api/chat`
  - Static file server at `/` for the React build

### `src/config.py` -- LLM Configuration

- Loads `GEMINI_API_KEY` from `.env`
- Exports `LLM_CONFIG` dict used by all AG2 agents:
  - Model: `gemini-3-flash-preview`
  - API type: `google`
  - Temperature: `0.2`
- Every agent in the system uses this same config. There is no per-agent model selection.

### `src/models.py` -- Pydantic Data Models

Defines the typed shapes of all data:

| Model | Purpose |
|-------|---------|
| `DocumentInput` | Request body for creating a document (`title`, `content`) |
| `ActionItem` | Single action: `task`, `owner`, `deadline`, `priority`, `dependencies` |
| `RiskItem` | Single risk: `description`, `category`, `severity` |
| `AnalysisResult` | Combined output: `summary`, `key_decisions`, `action_items[]`, `risks[]` |
| `DocumentResponse` | Document list item |
| `DocumentDetail` | Full document with analysis |
| `ChatRequest` | Chat endpoint input: `message`, optional `conversation_id` |

### `src/database.py` -- SQLite Connection

- Database file: `db/documents.db`
- `get_db()` returns an `aiosqlite` connection with WAL mode
- `init_db()` runs `db/schema.sql` to create tables

### `db/schema.sql` -- Database Schema

Four tables:

```
documents          -- Uploaded documents (title, content, word_count)
analyses           -- One per document (summary, actions, risks JSON, status)
conversations      -- Chat sessions (id, name, timestamps)
chat_messages      -- Messages within conversations (role, content, component, payload)
```

`analyses.status` tracks: `pending` -> `processing` -> `completed` / `failed`

`chat_messages.component` is either `null` (plain text) or `"analysis_card"` (rendered as a card in the UI). `payload` stores the full analysis JSON for card messages.

---

## The Two API Surfaces

### 1. REST Document API (`src/routes.py`)

This is a traditional CRUD API. It exists alongside (and independently of) the chat interface.

| Endpoint | What it does |
|----------|-------------|
| `POST /api/documents` | Saves a document to DB, kicks off analysis as a **background task** |
| `GET /api/documents` | Lists all documents with their analysis status |
| `GET /api/documents/{id}` | Returns full document + analysis result |
| `GET /api/analyses/{id}` | Returns just the analysis for a document |

**Flow:** POST creates the document row + an `analyses` row with status `"pending"`, then calls `_run_analysis()` as a FastAPI `BackgroundTask`. This calls `analyze_document()` (the non-streaming version in the orchestrator), updates the DB row when done.

This API is NOT used by the chat UI. It's a separate interface.

### 2. Chat SSE API (`src/chat_routes.py`)

This powers the chat interface. It uses Server-Sent Events for streaming progress.

| Endpoint | What it does |
|----------|-------------|
| `POST /api/chat` | Send a message, get streamed SSE response |
| `POST /api/chat/upload` | Upload .txt/.pdf, returns extracted text |
| `GET /api/chat` | List all conversations |
| `GET /api/chat/{id}/history` | Get message history for a conversation |

**Chat flow (POST /api/chat):**

```
1. Receive { message, conversation_id? }
2. Create or fetch conversation in DB
3. Save user message to chat_messages
4. Open SSE stream
5. Call route_message_streaming(message)
      |
      |--> classify_intent(message)     [LLM call #1]
      |       returns "conversation" or "analysis"
      |
      |--> IF conversation:
      |       simple_chat_streaming()   [LLM call #2]
      |       yields: { type: "result", component: null, full_reply: "Hi! ..." }
      |
      |--> IF analysis:
      |       analyze_document_streaming()  [3 LLM calls via GroupChat]
      |       yields: { type: "status", agent: "SummaryAgent" }
      |               { type: "agent_done", agent: "SummaryAgent" }
      |               { type: "status", agent: "ActionAgent" }
      |               { type: "agent_done", agent: "ActionAgent" }
      |               { type: "status", agent: "RiskAgent" }
      |               { type: "agent_done", agent: "RiskAgent" }
      |               { type: "result", component: "analysis_card", payload: {...} }
      |
6. chat_routes wraps the final "result" event into an "end" event:
      { type: "end", conversation_id, component, payload, full_reply }
7. Save assistant message to DB
```

**SSE event types the frontend receives:**

| Event | When | Frontend behavior |
|-------|------|-------------------|
| `status` | Agent starts working | Shows agent name in progress stepper |
| `agent_done` | Agent finished | Marks agent as complete in stepper |
| `end` | Final result ready | Adds message bubble to chat (text or analysis card) |
| `error` | Something broke | Shows error message |

---

## The Orchestrator (`src/orchestrator.py`) -- In Detail

This is the core of the system. It contains:

### `_quick_llm_call(system_message, user_message) -> str`

A utility that makes a single LLM call. Uses the same `UserProxyAgent.initiate_chat()` pattern as the analysis pipeline (creates a UserProxy + AssistantAgent, runs one turn, extracts the reply from chat history).

Used by: `classify_intent()` and `simple_chat_streaming()`.

### `classify_intent(message) -> "analysis" | "conversation"`

Asks the LLM to classify the user's message as one of two intents. The LLM is prompted to reply with a single word. If the reply contains "analysis", routes to the pipeline; otherwise defaults to conversation.

**This is not dynamic agent selection.** It's a binary if/else gate.

### `simple_chat_streaming(message)`

For conversational messages. Makes one LLM call with a friendly system prompt, yields a single result event with `component: null` (renders as plain text in the UI).

### `route_message_streaming(message)`

The entry point called by `chat_routes.py`. Calls `classify_intent()`, then delegates to either `simple_chat_streaming()` or `analyze_document_streaming()`.

### `analyze_document_streaming(document_text)`

The multi-agent analysis pipeline. Here's exactly what happens:

```
1. Create 3 AssistantAgents: SummaryAgent, ActionAgent, RiskAgent
2. Create 1 UserProxyAgent: Coordinator
3. Create a GroupChat with all 4 agents
     - speaker_selection_method = "round_robin"
     - max_round = 4  (Coordinator speaks once, then each agent once)
4. Coordinator sends the document as the opening message
5. AG2's GroupChat runs round_robin:
     Round 1: Coordinator sends document (already done)
     Round 2: SummaryAgent sees document, produces summary JSON
     Round 3: ActionAgent sees document + summary, produces action items JSON
     Round 4: RiskAgent sees document + summary + actions, produces risks JSON
6. Orchestrator polls group_chat.messages every 0.5s during execution
     - Yields SSE "status" and "agent_done" events as agents complete
7. After all agents finish, parses each agent's JSON from chat history
8. Builds AnalysisResult and yields it as the final "result" event
```

**Key point about round_robin:** The speaker order is FIXED. There is no dynamic selection, no "manager LLM" deciding who speaks next. AG2's GroupChat supports an `"auto"` speaker selection mode where an LLM decides the next speaker, but this system uses `"round_robin"` -- a hardcoded sequential order.

**Key point about agent context:** Each agent sees ALL prior messages in the GroupChat. So ActionAgent sees the original document AND SummaryAgent's output. RiskAgent sees everything before it. This is how they build on each other's work.

### `analyze_document(document_text) -> AnalysisResult`

Non-streaming version of the same pipeline. Used by the REST API's background task (`routes.py`). Same GroupChat logic, just returns the result directly instead of yielding SSE events.

### `_extract_json(text) -> dict | list | None`

Helper to parse JSON from agent responses. Handles:
- Raw JSON strings
- JSON wrapped in markdown code fences
- JSON embedded in surrounding text (finds first `{...}` or `[...]`)

---

## The Three Sub-Agents

All three are `AssistantAgent` instances (AG2's wrapper around an LLM with a system prompt). They have no tools, no code execution, no memory -- just a system prompt that tells them what JSON to produce.

### SummaryAgent (`src/agents/summary_agent.py`)

- **Input:** The raw document
- **Output JSON:** `{ "summary": "...", "key_decisions": ["...", "..."] }`
- Produces a 3-5 sentence summary preserving intent and constraints

### ActionAgent (`src/agents/action_agent.py`)

- **Input:** The raw document + SummaryAgent's output (via GroupChat history)
- **Output JSON:** `{ "action_items": [{ "task", "owner", "deadline", "priority", "dependencies" }] }`
- Extracts every actionable task, assignment, commitment, or next step

### RiskAgent (`src/agents/risk_agent.py`)

- **Input:** The raw document + SummaryAgent's output + ActionAgent's output
- **Output JSON:** `{ "risks": [{ "description", "category", "severity" }] }`
- Categories: `risk`, `open_question`, `assumption`, `missing_data`
- Identifies risks, unresolved questions, and assumptions

---

## LLM Calls Per User Message

| Scenario | LLM calls | What happens |
|----------|-----------|-------------|
| Simple chat ("Hello") | 2 | 1 for intent classification + 1 for the response |
| Document analysis | 4 | 1 for intent classification + 3 for the agent pipeline (Summary, Action, Risk) |
| REST API (POST /api/documents) | 3 | No classification, directly runs the 3-agent pipeline |

All calls go to `gemini-3-flash-preview` via AG2's Google Gemini integration.

---

## What This System Is NOT

- **Not a dynamic agent router:** There's no LLM deciding which agents to call based on content
- **Not a RAG system:** Documents are analyzed directly, not retrieved from a vector store
- **Not a multi-turn agent conversation:** Agents speak exactly once each in a fixed order
- **Not using agent tools/functions:** Agents only produce text (JSON), they don't call APIs or execute code
- **No conversation memory for the chat agent:** The simple chat handler gets only the current message, not prior conversation history
