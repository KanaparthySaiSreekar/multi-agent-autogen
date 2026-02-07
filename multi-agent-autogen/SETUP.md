# Setup Guide

## Prerequisites

| Tool | Minimum Version | Purpose |
|------|----------------|---------|
| Python | 3.10+ | Backend runtime |
| uv | any recent | Python package manager (faster pip alternative) |
| Node.js | 18+ | Frontend build tooling |
| npm | 8+ | Frontend package manager |
| Git | any | Version control |

You also need a **Google Gemini API key**. Get one at: https://aistudio.google.com/apikey

---

## Project Structure

```
multi-agent-autogen/
  db/
    schema.sql          # Table definitions (auto-run on startup)
    documents.db        # SQLite database (auto-created on startup)
  src/
    main.py             # FastAPI app entry point
    config.py           # LLM config (reads .env)
    database.py         # SQLite connection helper
    models.py           # Pydantic data models
    routes.py           # REST API for documents
    chat_routes.py      # Chat SSE API
    chat_models.py      # Chat request/response models
    orchestrator.py     # Agent pipeline + routing logic
    agents/
      summary_agent.py  # Produces document summary
      action_agent.py   # Extracts action items
      risk_agent.py     # Identifies risks
  frontend/
    src/                # React source code
    package.json        # Frontend dependencies
    vite.config.js      # Vite build config (outputs to ../static/)
  static/               # Built frontend (served by FastAPI)
    index.html
    assets/
  .env                  # API keys (not committed)
  pyproject.toml        # Python dependencies
  uv.lock               # Locked dependency versions
```

---

## Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd multi-agent-autogen
```

---

## Step 2: Set Up the Environment File

Create a `.env` file in the project root:

```bash
# Windows (PowerShell)
New-Item .env

# macOS/Linux
touch .env
```

Add your Gemini API key:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

Replace `your_gemini_api_key_here` with your actual key. This file is gitignored and will not be committed.

---

## Step 3: Install Python Dependencies

This project uses **uv** as the package manager.

### If you don't have uv installed:

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create virtual environment and install dependencies:

```bash
uv venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (cmd)
.venv\Scripts\activate.bat

# macOS/Linux
source .venv/bin/activate
```

Install all Python packages:

```bash
uv sync
```

This reads `pyproject.toml` and `uv.lock`, installing these packages:

| Package | Purpose |
|---------|---------|
| `ag2[gemini]` | AG2 multi-agent framework with Google Gemini support |
| `fastapi` | Web framework |
| `uvicorn` | ASGI server to run FastAPI |
| `aiosqlite` | Async SQLite driver |
| `python-dotenv` | Loads `.env` file |
| `pydantic` | Data validation |
| `pdfplumber` | PDF text extraction |
| `python-multipart` | File upload support |

---

## Step 4: Build the Frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

`npm run build` compiles the React app and outputs it to the `static/` directory. FastAPI serves these files directly -- there is no separate frontend server in production.

---

## Step 5: Start the Server

```bash
uvicorn src.main:app --reload
```

This will:
1. Start FastAPI on **http://localhost:8000**
2. Auto-create the SQLite database (`db/documents.db`) and tables on first run
3. Serve the React frontend at the root URL

Open **http://localhost:8000** in your browser.

---

## Frontend Development Mode (Optional)

If you're actively editing frontend code, you can run Vite's dev server for hot reload instead of rebuilding every time:

**Terminal 1 -- Backend:**
```bash
uvicorn src.main:app --reload
```

**Terminal 2 -- Frontend dev server:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173** (Vite's dev server). Vite is configured to proxy `/api` requests to `http://localhost:8000`, so the backend works seamlessly.

When you're done, run `npm run build` from the `frontend/` directory to update the production build in `static/`.

---

## Verify Everything Works

### 1. Check the app loads
Open http://localhost:8000 -- you should see the Document Intelligence chat UI.

### 2. Test casual conversation
Type "Hello" in the chat box. You should get a friendly text response (no agent progress stepper).

### 3. Test document analysis
Paste a block of text (meeting notes, a report, etc.) into the chat box. You should see:
- The 3-step agent progress indicator (Summary -> Actions -> Risks)
- An analysis card with summary, action items, and risks

### 4. Test file upload
Click the upload button, attach a `.txt` or `.pdf` file. The extracted text should appear in the chat input ready to send.

---

## Troubleshooting

### "GEMINI_API_KEY not set" or API errors
- Confirm `.env` exists in the project root (not in `src/` or `frontend/`)
- Confirm the key is valid at https://aistudio.google.com/apikey
- Restart the server after changing `.env`

### "Module not found" errors
- Make sure the virtual environment is activated (you should see `(.venv)` in your terminal prompt)
- Run `uv sync` again

### Frontend shows blank page
- Run `npm run build` from the `frontend/` directory
- Check that `static/index.html` exists and references a `.js` file in `static/assets/`

### Database errors
- The database auto-creates on startup. If it gets corrupted, delete `db/documents.db` and restart the server.

### Chat returns empty responses
- Check the server logs in the terminal for errors
- Confirm your Gemini API key has quota remaining

### Port already in use
```bash
# Use a different port
uvicorn src.main:app --reload --port 8001
```
If using a different backend port with the Vite dev server, update the proxy target in `frontend/vite.config.js`.

---

## Environment Summary

| Component | Technology | Port |
|-----------|-----------|------|
| Backend | FastAPI + Uvicorn | 8000 |
| Frontend (prod) | Static files served by FastAPI | 8000 |
| Frontend (dev) | Vite dev server | 5173 |
| Database | SQLite (file-based) | N/A |
| LLM | Google Gemini (`gemini-3-flash-preview`) via AG2 | N/A |
