# Dancelight RAG — Project Constitution

This file is the **single source of truth** for how work happens in this repo.
Claude Code reads this on every session. Humans should re-read it before
proposing changes. Conflicts between this file and conversation are resolved
in favour of this file.

---

## 1. What this project is

A hybrid (BM25 + BGE-M3 dense + bge-reranker + GPT-4o selection) RAG system
over the 2025 舞光 LED catalog (388-page PDF). Two entry points:

| Entry point | File | Purpose |
|---|---|---|
| Flask web app | `web/app.py` (port 8000) | Live search UI + GPT-4o pick + SQLite query logging |
| Research pipeline | `research/pipeline.py` | Full benchmark loop against `question.xlsx` (Qwen3-Embedding-8B + qwen3.6 via Ollama) |

Both share retrieval primitives from `rag/engine.py`.

---

## 2. Tech stack (pinned by `requirements.txt`)

- **Python** 3.10+
- **Web**: Flask, PyMuPDF (fitz)
- **Retrieval**: rank-bm25, jieba, sentence-transformers, torch (CUDA)
- **Embeddings**: BGE-M3 (web), Qwen3-Embedding-8B (research)
- **Reranker**: BAAI/bge-reranker-v2-m3 (cross-encoder)
- **LLM**: OpenAI `gpt-4o` (web final selection), Ollama `qwen3.6:latest` (research)
- **Storage**: SQLite (`dancelight_queries.db`), `.npy` for embeddings, JSON for annotation/image caches
- **Public URL**: localtunnel via cron-managed supervisor (see `lt/`)

Do not introduce a new framework without an approved spec.

---

## 3. Module boundaries (load-bearing — read carefully)

```
rag/                  # retrieval engine — no Flask, no SQLite, no I/O beyond cache files
  CONTRACT.md         # public API surface
  SUMMARY.md          # plain-language overview (auto-maintained)
  engine.py
  __init__.py

web/                  # Flask app + UI — depends on rag/, never reaches into its internals
  CONTRACT.md
  SUMMARY.md
  app.py
  templates/index.html
  __init__.py

research/             # benchmark + offline experiments — never imported by web/
  CONTRACT.md
  SUMMARY.md
  pipeline.py
  __init__.py
```

**Hard rule**: changing module A **must not change** the externally observable
behavior of module B. Module B's behavior is defined by `B/CONTRACT.md` + its
contract tests. If you need to change another module's contract, that is a
**spec-level change** — write a spec.md first.

---

## 4. The workflow (mandatory for every feature / non-trivial change)

```
1. spec.md       ─►  2. user approval  ─►  3. CONTRACT update  ─►
4. contract test (red)  ─►  5. implementation  ─►  6. contract test (green)  ─►
7. SUMMARY.md update (auto via hook)  ─►  8. commit
```

### 4.1 Spec first

Before writing code, draft `specs/<feature-slug>.md` using
`docs/templates/SPEC_TEMPLATE.md`. Three sections, all required:

- **目標 (Goal)** — one paragraph, plain language
- **約束 (Constraints)** — what we may not break / change (other modules' contracts, latency, cost, dependencies)
- **成功條件 (Success criteria)** — verifiable, ideally as test names

Do not start implementation until the user approves the spec. "Approved" means
the user replies with explicit "go" / "approved" / "可以動手" or equivalent.

### 4.2 Contract changes are spec-gated

If a change modifies any function signature, return shape, HTTP endpoint
path/method/body shape, or any behavior another module relies on, the spec
must call this out explicitly and the corresponding `CONTRACT.md` must be
updated in the same change.

### 4.3 Tests are the truth

A change is **not done** until `pytest tests/contract/ -q` is green. CC may
not declare a task complete with red tests. If a contract test must change
to accommodate the spec, the spec must say so.

### 4.4 SUMMARY is auto-maintained

After Edit/Write in any module, a PostToolUse hook reminds CC to refresh
that module's `SUMMARY.md`. CC must comply before ending the turn — these
files are how humans (and future CC sessions) quickly understand the code.

---

## 5. Prohibited actions

- ❌ Reaching into another module's private state (`web/` may not touch
  `rag.engine._llm_select` directly; use `rag.engine.search`).
- ❌ Skipping spec.md for non-trivial changes (>1 file or any contract change).
- ❌ Marking a task complete with failing contract tests.
- ❌ Editing `CONTRACT.md` without updating the spec.md that motivated it.
- ❌ Adding production code under `research/` or vice versa.
- ❌ Committing with `--no-verify`, `--no-gpg-sign`, or amending past commits.
- ❌ Force-pushing to `main`.
- ❌ Adding new top-level scripts at repo root for feature code (only
  configuration, entry shims, and infra scripts allowed at root).

---

## 6. Conventions

- **Comments**: WHY only, not WHAT. Names should already say WHAT.
- **Logging**: print to stdout with `[<module>]` prefix (e.g. `[web] starting...`).
- **Errors**: don't catch what you can't handle. Let it propagate to Flask's
  500 handler or the pipeline's per-question failure log.
- **Secrets**: `.env` only. Never committed. `.env.example` lists keys.
- **Embedding cache files**: regenerated on startup if missing. Delete to force rebuild.
- **Chunk text immutability**: `annotations_cache.json` is keyed by
  `md5(chunk_text)[:16]`. Do not change chunk text generation without
  invalidating the cache (and noting in the spec).

---

## 7. Running

```bash
# Web app (production-ish)
cd /root/dancelight-rag-repo
python3 -m web.app                 # serves :8000

# Research pipeline (benchmark)
python3 -m research.pipeline

# Contract tests (must pass before declaring any task done)
pytest tests/contract/ -q
```

Public URL (stable, via localtunnel + cron supervisor):
**https://dancelight-rag-nccu.loca.lt**

---

## 8. Out of scope of this file

- Deployment / scaling beyond single-host
- Multi-user auth (currently anonymous, IP-only logging)
- CI (not yet set up; tests run locally and on-demand)
