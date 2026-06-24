# BrewMatch — Web App

Live demo of the pipeline. Type a craving, watch the three stages animate, get a ranked list.

## Local development

```bash
# 1. Install backend deps (from repo root)
pip install -r web/requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Start the server
uvicorn web.backend.main:app --reload --port 8000
```

Then open http://localhost:8000.

## Deploy to Render (one-click)

1. Push this repo to GitHub (already done).
2. New Web Service → connect your repo.
3. Set **Root directory**: *(leave blank)*
4. Set **Build command**: `pip install -r web/requirements.txt`
5. Set **Start command**: `uvicorn web.backend.main:app --host 0.0.0.0 --port $PORT`
6. Add environment variable: `ANTHROPIC_API_KEY` = your key.
7. Deploy.

## Architecture

```
browser  →  GET /          → frontend/index.html  (static)
browser  →  POST /api/rank → backend/main.py
                              ├─ stage1_extract.py  (Claude)
                              ├─ stage2_filter.py   (pandas)
                              └─ stage3_rank.py     (sentence-transformers)
```

Rate limiting: 10 requests / 60s per IP (in-memory).  
Caching: identical queries (case-insensitive) are served from memory — no duplicate Claude calls.
