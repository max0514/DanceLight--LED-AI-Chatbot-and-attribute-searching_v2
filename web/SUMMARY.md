# `web/` — Summary

**Auto-maintained.** Refresh after any non-trivial change to `app.py` or `templates/`.

---

## What this module does

A thin Flask app that serves a one-page search UI and exposes 4 JSON
endpoints + a PDF page-image endpoint. Every query and (optional) follow-up
feedback is logged to a local SQLite DB for later analysis. Heavy lifting
is delegated to `rag.engine`; this module only handles HTTP, templating,
and persistence.

## Key files

| File | Purpose |
|---|---|
| `app.py` | Flask routes, SQLite init, PDF page renderer, entry-point `__main__`. |
| `templates/index.html` | Single-page UI: search box, "★ 推薦 + 4 備選" cards, PDF page modal, tooltips. |
| `CONTRACT.md` | HTTP API + DB schema (stable; spec-gated to change). |

## How to use it (from outside)

```bash
# Start
cd /root/dancelight-rag-repo && python3 -m web.app
# Or let cron keep it alive: /etc/cron.d/dancelight-web

# Query
curl -X POST http://127.0.0.1:8000/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"15W崁燈 6500K"}'
```

Production URL: `https://dancelight-rag-nccu.loca.lt` (loca.lt tunnel).

## Internal notes (gotchas)

- **CWD matters**: `DB_PATH = "./dancelight_queries.db"` and `rag.engine` paths are relative. Always launch from repo root.
- **Conda Python required**: deps (`fitz`, `sentence-transformers`, `openai`) live in `/opt/conda/envs/py_3.10/`. The cron ensure-script hardcodes this path; system `python3` will `ModuleNotFoundError`.
- **Page-image cache is unbounded**: each `(page, dpi)` PNG stays in memory forever. 388 pages × a few DPIs = manageable, but restart resets it.
- **CF-Connecting-IP**: real client IP comes through the tunnel/Cloudflare layer; `request.remote_addr` would be 127.0.0.1.
- **Tooltip / modal logic is in `index.html` JS**: not server-rendered. Card layout is responsive via `clamp()`.
- **DB write failures are non-fatal**: search still returns to user; only `query_id` becomes null. Logged to stdout with `[db]` prefix.

## Last updated

2026-05-29 — moved from root `dancelight_web_app.py` into `web/` package, added contract + summary.
