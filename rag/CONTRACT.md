# Contract: `rag/`

Public retrieval/selection API. Callers (`web/`, `research/`, tests) may rely
on everything documented here. Anything else is implementation detail.

Changes to this file require a corresponding `specs/<feature>.md`.

---

## Public API

### `rag.engine.initialize() -> None`

Build chunks + BM25 index, load cached BGE-M3 embeddings, lazy-init reranker
on first `search()` call. **Idempotent** — safe to call repeatedly; subsequent
calls are near-zero cost.

**Side effects**: reads `PDF_PATH`, `ODL_JSON`, `IMG_CACHE_FILE`,
`EMBED_CACHE`, `ANNOTATIONS_CACHE` (paths below). If `EMBED_CACHE` is missing,
embeds all chunks (slow — minutes on GPU, hours on CPU).

**Raises**: `FileNotFoundError` if `PDF_PATH` or `ODL_JSON` is missing.

---

### `rag.engine.search(query: str, top_k: int = 5) -> list[dict]`

End-to-end pipeline: hybrid retrieve → BGE rerank → GPT-4o select.

**Inputs**:
- `query` — natural-language Chinese product spec (e.g. `"15W崁燈 6500K"`).
- `top_k` — number of picks. **Currently only `5` is supported** (LLM prompt
  produces 1 recommendation + 4 alternates). Other values may work but are
  not contractually guaranteed.

**Returns**: list of exactly 5 dicts (or fewer on engine failure), shape:

```python
{
    "rank_label": "★ 推薦" | "備選 1" | "備選 2" | "備選 3" | "備選 4",
    "name": str,                       # product series name (LLM-extracted, defensively cleaned)
    "category": str,                   # e.g. "崁燈", "投射燈"
    "page": int,                       # 1-indexed PDF page number
    "models": str,                     # comma-separated model codes
    "wattages": str,                   # comma-separated, e.g. "15"
    "color_temps": str,                # comma-separated K values
    "lumens": str,
    "ip_rating": str,
    "features": str,                   # comma-separated tags
    "score": float,                    # LLM-assigned 0.0–1.0 confidence
    "reason": str,                     # LLM-generated rationale (one short sentence)
}
```

**Side effects**: one OpenAI API call to `LLM_SELECT_MODEL`. ~25–35s p50 latency.

**Raises**: propagates OpenAI exceptions; callers should treat as 5xx.

---

## Public constants

- `LLM_SELECT_MODEL: str` — currently `"gpt-4o"`. The model used for final selection.
- `LOCAL_LLM: str` — alias for `LLM_SELECT_MODEL`. Used by UI label.
- `PDF_PATH: str` — `"./2025舞光LED21st(單頁水印可搜尋).pdf"`. The catalog source.
- `EMBED_MODEL: str` — `"BAAI/bge-m3"`.
- `RERANK_MODEL: str` — `"BAAI/bge-reranker-v2-m3"`.

The path constants assume CWD is the repo root. Callers must `cd` there or
adjust their environment.

---

## Invariants

1. `initialize()` is idempotent.
2. `search()` is safe to call without explicit `initialize()` — it self-initializes on first use.
3. `search()` returns at most `top_k` dicts; never raises for empty result.
4. `name` field is post-processed (`_is_bad_name` blacklist + `_fallback_name`) — guaranteed not to be a warning line, marketing slogan, list prefix, or "第N頁產品" placeholder.
5. `page` is 1-indexed (matches the PDF's printed page numbers).
6. The `annotations_cache.json` md5 keys are derived from chunk text — **do not modify chunking logic without rebuilding the cache**.

---

## Backwards compatibility

**BREAKING changes** (require spec.md + bump):
- Removing or renaming `initialize`, `search`, or any public constant.
- Changing `search()` return dict keys or value types.
- Changing the `rank_label` enum values.
- Changing the LLM provider or model in a way that alters output shape.

**NON-BREAKING** (safe in minor changes):
- Adding new fields to the result dict.
- Internal refactor of retrieval weights, rerank thresholds, candidate pool size.
- Swapping cache file format (as long as `initialize()` handles both).
- Improving `name` extraction accuracy.

---

## Dependencies on other modules

`rag/` depends on:
- The repo root being CWD (for relative paths).
- `OPENAI_API_KEY` env var (loaded via `python-dotenv` from `.env`).
- The PDF + `output_opendataloader/` + cache files being present.

`rag/` **must not** depend on `web/` or `research/`.
