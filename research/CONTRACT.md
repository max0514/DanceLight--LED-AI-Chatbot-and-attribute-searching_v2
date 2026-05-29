# Contract: `research/`

**`research/` is a script, not a library.** It is the offline benchmark
pipeline: load the 17 test queries from `question.xlsx`, run the full RAG +
local LLM (qwen3.6) flow, score against `answer.xlsx`, write
`rag_results.xlsx`. It is independent of `web/` and uses its own embedding
model (Qwen3-Embedding-8B, not BGE-M3).

The "contract" here is about inputs, outputs, and config knobs — not Python
function signatures (callers do not import from `research/`).

Changes to this file require a corresponding `specs/<feature>.md`.

---

## Entry point

```bash
cd /root/dancelight-rag-repo && python3 -m research.pipeline
```

Runs to completion. No CLI arguments. Reads config from module-level constants.

---

## Inputs (must exist before running)

| Path | Format | Purpose |
|---|---|---|
| `./question.xlsx` | xlsx | 17 test queries, one per row, column `query` |
| `./answer.xlsx` | xlsx | Gold model numbers per query for scoring |
| `./2025舞光LED21st(單頁水印可搜尋).pdf` | PDF | Catalog source (same as `web/`) |
| `./output_opendataloader/` | dir | Pre-extracted structural JSON + images |
| `./qwen3_8b_embeddings/chunk_embeddings.npy` | npy | Cached embeddings (rebuilt if missing — slow) |
| `./annotations_cache.json` | json | Per-chunk annotations (rebuilt if missing — slow + costs $) |
| Ollama daemon | service | Local LLM (default model `qwen3.6:latest`) running on `:11434` |

---

## Outputs

| Path | Format | Description |
|---|---|---|
| `./rag_results.xlsx` | xlsx | One row per question with retrieved docs, LLM answer, hit/miss flags |
| stdout | text | Per-question status + summary metrics at the end |

The summary line format is part of the contract (parsed by external scripts):
```
檢索命中率 (gold in top-20):  N/M = X.X%
LLM 任一選項命中 (acc_any):   N/M = X.X%
LLM 推薦命中     (acc_rec):   N/M = X.X%
```

These three metrics are stable. Renaming or restructuring requires a spec.

---

## Configuration (module-level constants)

| Const | Default | Meaning |
|---|---|---|
| `EMBED_MODEL` | `"Qwen/Qwen3-Embedding-8B"` | Dense embedder (4096-dim, fp16) |
| `RERANK_MODEL` | `"BAAI/bge-reranker-v2-m3"` | Cross-encoder reranker |
| `LOCAL_LLM` | `"qwen3.6:latest"` | Ollama model tag for answer generation |
| `TOP_K` | 20 | Reranked candidates passed to LLM |
| `RETRIEVE_K` | 50 | Hybrid retrieval pool size |
| `BM25_WEIGHT` | 0.5 | Hybrid weight for BM25 score |
| `VECTOR_WEIGHT` | 0.5 | Hybrid weight for dense score |
| `CTX_MAX_CHARS` | 40000 | Total context window for LLM prompt |

Changing any of these in a way that meaningfully affects benchmark scores
requires a spec.md noting the prior baseline and the new result.

---

## Invariants

1. **Reads from CWD = repo root.**
2. **Does not mutate state outside this module + its output files.** No DB writes, no network logging.
3. **Independent of `web/`.** Restarting Flask must not affect a running benchmark.
4. **Per-question failure does not abort the run** — failures are logged and the harness continues.
5. **The 17 test queries are the canonical eval set.** Adding/removing queries requires updating `question.xlsx` + `answer.xlsx` together + a spec.

---

## Backwards compatibility

**BREAKING** (spec-gated):
- Changing the `rag_results.xlsx` column schema.
- Changing summary stdout format.
- Changing `EMBED_MODEL`, `RERANK_MODEL`, or `LOCAL_LLM` (affects scores).
- Removing `question.xlsx` / `answer.xlsx`.

**NON-BREAKING**:
- Internal refactors that preserve output xlsx + stdout summary.
- Adding new columns to `rag_results.xlsx`.
- Adding new diagnostic prints.

---

## Dependencies on other modules

`research/` **must not** import from `web/` or `rag/`. It is intentionally a
duplicate implementation — they have diverged (different embedder, different
LLM, different scoring) and unifying them would couple benchmark drift to
production behavior.

If you want a shared utility, propose it via spec.md first.
