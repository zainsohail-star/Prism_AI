# PRISM AI — Precise Retrieval & Intelligent Search Machine

A three-lens question-answering system built with LangChain, LangGraph, FastAPI, and Groq.

Like a real prism splits one beam of light into a spectrum, PRISM AI splits one question into three lenses — your document, your document plus AI knowledge, or the live web — so you always get an answer from the angle that fits.

## Project Structure

```
prism-ai/
├── capstone.ipynb     # Teaching notebook — run this first to understand the pipeline
├── ingestion.py       # CLI: embed a PDF and save a FAISS index
├── rag.py             # Module: get_answer(question) → {answer, sources}
├── agent.py            # LangGraph web-search agent (Lens 3)
├── spectrum.py         # LangGraph full-spectrum agent — runs & merges all 3 lenses (Lens 4)
├── main.py            # FastAPI server — four endpoints
├── index.html         # Single-file chat UI
├── requirements.txt
├── .env               # API keys (never commit this)
└── deployment.md      # Railway + Vercel instructions
```

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env` and fill in your keys:

```
GROQ_API_KEY=gsk_...
TAVILY_API_KEY=tvly-...
```

## Usage

### Option A — CLI ingestion (then run the API)

```bash
# 1. Ingest a PDF
python ingestion.py --file your_document.pdf

# 2. Start the API
uvicorn main:app --reload

# 3. Open index.html in a browser (double-click or use Live Server)
```

### Option B — Upload via the UI

Start the API first, then upload a PDF directly from the sidebar in `index.html`.

## Three Lenses

| Lens | What it does |
|------|-------------|
| **1 — Document Only** | Answers strictly from the uploaded PDF. No outside knowledge. |
| **2 — Document + AI Knowledge** | Uses the document first; supplements with general knowledge when the document is insufficient. Prefixes additions with "From general knowledge:". |
| **3 — Web Search** | Ignores any local document. Uses Tavily to fetch live search results, then answers with Groq. |
| **4 — Full Spectrum** | Runs all three lenses at once (document, general knowledge, web search) through a LangGraph pipeline, then a synthesis step cross-checks them and merges them into one answer — flagging it if the lenses disagree. The UI lets you expand "View by lens" to see what each one said individually. |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/index-status` | Returns `{"indexed": true/false}` |
| `POST` | `/ingest` | Upload a PDF (multipart form), returns `{"status":"ok","chunks":N}` |
| `POST` | `/ask` | `{"question": str, "mode": 1\|2\|3\|4}` → `{"answer": str, "sources": [...], "lenses": {...} (mode 4 only)}` |

## Tech Stack

- **LLM** — Groq (`llama-3.3-70b-versatile`)
- **Embeddings** — HuggingFace `all-MiniLM-L6-v2` (local, no API key needed)
- **Vector store** — FAISS (local disk)
- **Agent** — LangGraph (two-node graph for web search)
- **Web search** — Tavily
- **API** — FastAPI + Uvicorn
- **Frontend** — Vanilla HTML/CSS/JS (no build step), themed around the PRISM spectrum: violet for Document Only, cyan for Document + AI, amber for Web Search.

### Frontend power features
- **Markdown-rendered answers** — bold text, lists, and code blocks from the model render properly instead of showing as raw text.
- **Response time readout** — every answer shows how long it took (e.g. `1.4s`), so slow calls are visible at a glance.
- **Copy** — one click to copy any answer to the clipboard.
- **Regenerate** — re-ask the last question on demand without retyping it.
- **New Chat** — clears the conversation and starts fresh.
