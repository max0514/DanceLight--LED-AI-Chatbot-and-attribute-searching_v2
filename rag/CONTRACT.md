# Contract: `rag/`

Public retrieval/selection API. Callers (`web/`, `research/`, tests) may rely
on everything documented here. Anything else is implementation detail.

Changes to this file require a corresponding `specs/<feature>.md`.

---

## Public API

### `rag.engine.initialize(edition: str | None = None) -> None`

Build chunks + BM25 index, load cached BGE-M3 embeddings, lazy-init reranker
on first `search()` call. **Idempotent per edition** — safe to call
repeatedly; subsequent calls are near-zero cost.

`edition=None` (default) loads every edition listed in `CORPUS` — used by the
web app at startup so both 21st and 22nd are warm. A specific string
(`"21st"` or `"22nd"`) loads just that one — used by `search()`'s lazy path.

**Side effects**: reads each edition's `pdf` + `odl_json`, plus the shared
`IMG_CACHE_FILE` and `ANNOTATIONS_CACHE`. If an edition's `embed_cache` is
missing, embeds all of that edition's chunks (slow — minutes on GPU, hours
on CPU). The annotations cache is shared across editions (md5-keyed by chunk
text).

**Raises**: `FileNotFoundError` if a corpus's PDF or ODL JSON is missing.

---

### `rag.engine.search(query: str, top_k: int = 5, *, edition: str = "22nd") -> list[dict]`

End-to-end pipeline: hybrid retrieve (BM25 + dense, top-100) →
**strict-scored GPT-4o rerank** of top-30 → returns top-5. Graduated 2026-06-10
from `experiment/engine.py`'s `strict30/pure` config; outperforms the prior
BGE-cross-encoder + LLM-select path on question.xlsx (9/15 vs 4/15 hit@5).

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
    "score": float,                    # hybrid-retrieve score (0.0–1.0); not LLM-assigned
    "reason": str,                     # LLM-generated rationale (≤25 字)
    "llm_breakdown": str,              # NEW (2026-06-10): per-dimension score breakdown from strict rerank, e.g. "類別10 瓦數8 IP10 ..."
}
```

**Side effects**: one OpenAI API call to `LLM_SELECT_MODEL`. ~25–35s p50 latency.

**Raises**: propagates OpenAI exceptions; callers should treat as 5xx.

---

## Public constants

- `LLM_SELECT_MODEL: str` — currently `"gpt-4o"`. The model used for final selection.
- `LOCAL_LLM: str` — alias for `LLM_SELECT_MODEL`. Used by UI label.
- `DEFAULT_EDITION: str` — currently `"22nd"`. The edition used when `search()` is called without an explicit `edition`.
- `CORPUS: dict[str, dict]` — mapping of edition key → `{label, pdf, odl_json, embed_cache}`. Iterate `CORPUS.items()` to enumerate available editions (e.g. for a UI selector).
- `PDF_PATH: str` — `CORPUS[DEFAULT_EDITION]["pdf"]`. Backward-compat shim; new code should read `CORPUS[edition]["pdf"]` instead.
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
- Removing an edition from `CORPUS` (callers may hard-code keys).

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
