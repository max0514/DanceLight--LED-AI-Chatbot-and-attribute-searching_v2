# Spec: graduate strict30/pure LLM rerank to rag/engine.py

**Status**: draft
**Author**: Claude  **Date**: 2026-06-10
**Affected modules**: rag / web

---

## 目標 (Goal)

`experiment/engine.py` 在 question.xlsx 上跑出 9/15 hit@5 (60% raw,75%
corpus-reachable) 的 `strict30/pure` 配置已驗證優於 production v7
(4/15 = 27%)。把這條 pipeline 直接搬到 `rag/engine.py`,讓 web app 預設
就用它。對外 API 完全不變;只是內部更換 retrieve→rerank→select 的具體做法。

---

## 約束 (Constraints)

- [ ] `rag/CONTRACT.md` — `search(query, top_k=5, *, edition='22nd')` signature
      跟 return dict shape 不變。新增 optional `llm_breakdown` 欄位 (NON-BREAKING)
- [ ] `web/app.py` 不改 — 透過 `rag_engine.search()` 取得結果即可
- [ ] `web/CONTRACT.md` 不改 — JSON shape 同 today
- [ ] 延遲預算 — strict 路徑跑 30 候選 (vs 20),latency 上限應仍 < 30s p95
- [ ] 成本 — 每次查詢仍 1 次 GPT-4o call (input ~12K tokens vs ~10K today,
      output ~600 tokens) — 約 1.5× 但量級不變
- [ ] Backward-compat — 同樣 chunks / embeddings / metadata,無需重建任何 cache

---

## 成功條件 (Success criteria)

- [ ] `rag.engine.search("15W崁燈 6500K")` 回 5 個結果且 `reason` 欄為
      strict prompt 產生的中文短句 (≤25 字)
- [ ] 每個結果含 `llm_breakdown` 欄 (類別/瓦數/IP 分項分數),未來 UI 可顯示
- [ ] webapp 端到端: `/api/search` 仍回正常 shape;UI cards 顯示 reason
- [ ] 21 版 / 22 版 search 皆通過 `scripts/smoke_test.py`

---

## Non-goals

- 重跑 question.xlsx 對齊 22 版 corpus (eval set 仍只對齊 21 版,等舞光 ground truth)
- Ensemble / multi-judge — strict30/pure 已最佳單通道,不加 ensemble
- 修改 hybrid retrieve / metadata filter / 同義詞 — 仍用既有 RRF + 1.3/0.7 boost
- 從 rag/engine.py 移除舊的 `_llm_select` / `LLM_SELECT_PROMPT` — 留著做 A/B,
  search() 不再調用即可

---

## 設計 (Design)

`search()` 流程從:
```
hybrid_retrieve(k=50) → bge_rerank(top 20) → llm_select(LLM_SELECT_PROMPT, 20→5)
```
改為:
```
hybrid_retrieve(k=100) → strict_llm_rerank(STRICT_RERANK_PROMPT, 30→5)
                                            (skip BGE rerank)
```

### 新增 constants
- `RETRIEVE_K = 100` (was 50) — strict 路徑要更多候選
- `STRICT_RERANK_N = 30` — LLM 看 30 個 (vs 20)
- 保留 `LLM_SELECT_CANDIDATES = 20` 給舊 path (不再呼叫但留著)

### 新增 / 移植函式
- `STRICT_RERANK_PROMPT` — 整段照搬 `experiment/engine.py:350`
- `_strict_llm_rerank(query, specs, candidates, top_k, n)` — 照搬 `experiment/engine.py:389`
  - 但用 `rag.engine._build_llm_context()`,`_get_openai()`,跟既有的 _llm_select 共享

### 移除 BGE rerank 路徑
- `_bge_rerank` 函式仍留著 (給 experiment/engine.py 用),但 `search()` 不再 call
- `_get_reranker()` 與 reranker model loading 是 lazy,沒人 call 就不會載入,
  webapp 啟動成本下降 ~2GB GPU 記憶體

### search() 的 result 多帶 `llm_breakdown`
原本只有 `reason`(LLM 推薦理由)。新增 `llm_breakdown` (例如「類別10 瓦數8
IP10 完整度6」)。CONTRACT 標 NON-BREAKING 新增欄位。

---

## 風險與緩解

- **風險**: 22 版 corpus 上 strict 表現未驗證,可能不如預期。
  **緩解**: 21 版 corpus 跑 smoke_test 確認;22 版 corpus 等舞光 ground truth。
  若回報差,改 `search()` 一行(把 search() 切回舊 path) 即可回滾。
- **風險**: BGE reranker 不再被 production load,但 experiment/engine.py 仍會 lazy load。
  **緩解**: 兩者隔離,沒問題。webapp 不主動 load。
- **風險**: 較大的 retrieve_k + LLM context 可能拉高 p95 latency。
  **緩解**: 預期 ~25-30s p50 (vs today ~22s)。若超 60s 考慮降 strict_n。
- **風險**: GPT-4o 偶爾回 JSON 結構不符 (picks 空 / wrong shape)。
  **緩解**: `_strict_llm_rerank` 已有 fallback — pad 用 hybrid 順序。
