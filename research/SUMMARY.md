# `research/` — Summary

**Auto-maintained.** Refresh after any non-trivial change to `pipeline.py`.

---

## What this module does

A reproducible benchmark harness. Given 17 fixed test queries
(`question.xlsx`), runs the full RAG pipeline using Qwen3-Embedding-8B +
bge-reranker + local qwen3.6 LLM (via Ollama), then scores each answer
against gold-standard model numbers in `answer.xlsx`. Outputs per-query
results to `rag_results.xlsx` and three aggregate accuracy metrics to stdout.

This module is **offline, single-shot**. Run it after tuning to measure
whether changes helped or hurt.

## Key files

| File | Purpose |
|---|---|
| `pipeline.py` | The whole benchmark — chunking, retrieval, rerank, LLM call, scoring. Notebook-style: top-to-bottom executable. |
| `hyde_llm_rerank_experiment.py` | One-off ablation (2026-06-08): HyDE × LLM-rerank on `Training.xlsx`. See `specs/hyde-llm-rerank-experiment.md` for results. **Conclusion: don't adopt.** |
| `rerank_metadata_tuning_experiment.py` | One-off ablation (2026-06-08): rerank-query enrichment × metadata-filter strength on `question.xlsx` (17 Q). See `specs/rerank-query-and-metadata-filter-tuning.md`. **Conclusion: adopt C (enrich rerank query); skip B/D.** |
| `CONTRACT.md` | Inputs / outputs / config (stable; spec-gated). |

## How to use it (from outside)

```bash
cd /root/dancelight-rag-repo
ollama serve &                            # if not running
python3 -m research.pipeline              # ~5-15 min on GPU; longer on CPU
cat rag_results.xlsx                      # inspect per-question results
# Summary metrics printed at the end of stdout.
```

## Internal notes (gotchas)

- **Different stack from `web/`**: uses Qwen3-Embedding-8B (4096-dim) vs. web's BGE-M3 (1024-dim). They are intentionally separate caches — don't try to share.
- **Local LLM (qwen3.6 via Ollama)**: requires `ollama serve` running. If Ollama is down, every query fails — but the harness keeps going. Look for "[ollama]" errors in stdout.
- **No `if __name__ == "__main__"` guard**: importing this module runs the benchmark. That's why it's `python3 -m research.pipeline`, not `from research import pipeline`.
- **Annotation cache is shared with `web/`**: same `./annotations_cache.json`. Both modules respect the md5-keyed chunk_text invariant.
- **Notebook origin**: file was converted from .ipynb, so there are empty cells at the end (`# In[ ]:`). Harmless; don't bother cleaning unless touching the file anyway.
- **v3.2 (current) reverts multimodal LLM**: GPT-4o + image URLs was worse (acc_any 41.2% → 17.6%); we're text-only. Don't reintroduce without spec + new baseline.

## Last updated

2026-06-08 — added `rerank_metadata_tuning_experiment.py` (rerank-query enrichment; C variant adopted into engine).
2026-06-08 — added `hyde_llm_rerank_experiment.py` (HyDE ablation; not adopted).
2026-05-29 — moved from root `dancelight_rag.py` into `research/` package, added contract + summary.
