# PRISM AI — Deployment Guide

## 1. Local Run

```bash
# Terminal 1 — start the API
uvicorn main:app --reload
# API is now at http://localhost:8000

# Terminal 2 (optional) — verify
curl http://localhost:8000/health
```

Open `index.html` by double-clicking it, or use VS Code Live Server.
The API URL in the HTML is already set to `http://localhost:8000`.

---

## 2. Deploy the FastAPI backend to Railway

### Step 1 — Push your code to GitHub
Make sure your repo contains: `main.py`, `agent.py`, `rag.py`, `ingestion.py`, `requirements.txt`.
Do **not** commit `.env` or the `faiss_index/` folder.

### Step 2 — Create a Railway project
1. Go to [railway.app](https://railway.app) and sign in.
2. Click **New Project** → **Deploy from GitHub repo** → select your repo.
3. Railway auto-detects Python and runs `pip install -r requirements.txt`.

### Step 3 — Set environment variables
In Railway → your service → **Variables**, add:

| Key | Value |
|-----|-------|
| `GROQ_API_KEY` | `gsk_...` |
| `TAVILY_API_KEY` | `tvly-...` |

### Step 4 — Set the start command
In Railway → your service → **Settings** → **Start Command**:

```
uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Step 5 — Get your public URL
Railway gives you a URL like `https://your-app.up.railway.app`.
Copy it — you will need it for the frontend.

> **Note on FAISS index:** Railway's filesystem is ephemeral.
> Upload a PDF via the `/ingest` endpoint after each deployment,
> or mount a persistent volume and set `INDEX_PATH` to that path.

---

## 3. Deploy the frontend to Vercel

### Step 1 — Change the API URL (the one line you must edit)

Open `index.html` and find this line near the top of the `<script>` block:

```js
// ── CHANGE THIS before deploying to Vercel ──
const API_URL = "http://localhost:8000";
```

Replace the value with your Railway URL:

```js
const API_URL = "https://your-app.up.railway.app";
```

### Step 2 — Deploy to Vercel
1. Go to [vercel.com](https://vercel.com) and sign in.
2. Click **Add New** → **Project**.
3. Drag and drop your `index.html` file, **or** import from GitHub.
4. Vercel deploys it instantly — no build step required.
5. Your chat UI is live at `https://your-project.vercel.app`.

---

## Quick Reference

| What | Command / URL |
|------|--------------|
| Local API | `uvicorn main:app --reload` |
| Local UI | Open `index.html` in browser |
| Railway start command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
| Railway env vars | `GROQ_API_KEY`, `TAVILY_API_KEY` |
| One line to change in HTML | `const API_URL = "..."` |
