# `rag/` — Summary

**Auto-maintained.** Refresh after any non-trivial change to `engine.py`.

---

## What this module does

Given a natural-language Chinese product spec (e.g. "15W崁燈 6500K"), returns
the top-5 matching products from the 388-page 舞光 LED catalog, with a
GPT-4o-generated rationale per pick. The pipeline is BM25 + BGE-M3 dense
hybrid retrieval → bge-reranker cross-encoder → GPT-4o final selection.

## Key files

| File | Purpose |
|---|---|
| `engine.py` | The whole pipeline. `initialize()` builds chunks + indexes; `search(query)` runs end-to-end. |
| `__init__.py` | Re-exports `engine` as `rag.engine`. |
| `CONTRACT.md` | Public API surface (stable; spec-gated to change). |

## How to use it (from outside)

```python
from rag import engine as rag_engine

rag_engine.initialize()                          # optional — search() self-inits
results = rag_engine.search("15W崁燈 6500K", top_k=5)
for r in results:
    print(r["rank_label"], r["name"], r["page"], r["reason"])
```

## Internal notes (gotchas)

- **Chunk text immutability**: `annotations_cache.json` keys are `md5(chunk_text)[:16]`. Any change to chunking logic invalidates the cache → annotations must be regenerated (slow + ~$0.50 OpenAI). Spec must call this out.
- **GPT-4o is the bottleneck**: p50 latency is ~25–35s, dominated by LLM call. Hybrid retrieve + rerank are <3s.
- **Name extraction is defense-in-depth**: GPT-4o is asked to extract `name`; we then filter (`_is_bad_name`) for warning lines / marketing slogans / placeholder patterns; fall back to chunk re-scan if needed. Don't simplify the cascade — each layer caught real bugs.
- **Dedup before LLM**: `_dedup_candidates` strips trailing `-XX` variant suffix from first model code. Prevents adjacent-page near-duplicates (e.g. p.138 D-CEC24DSW-LW vs p.139 D-CEC24DSW) from both winning slots.
- **Reranker uses 2000-char truncation**: long docs lose tail context. Acceptable for catalog-style chunks that put model codes early.
- **CWD must be repo root**: all `./xxx` paths are relative. `python3 -m web.app` from anywhere else will fail.

## Last updated

2026-05-29 — initial extraction from monolithic `rag_engine.py` into module.
