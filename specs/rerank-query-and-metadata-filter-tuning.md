# Spec: BGE-rerank query enrichment + metadata-filter strength tuning

Status: **EXPERIMENT COMPLETE — adopt C, awaiting approval to modify rag/engine.py**
Outcome (2026-06-08): C.enrich-rerank crossed the +2 threshold (+3 hits, 0 losses on n=15).
Owner: Claude Code (research → engine if effective)
Created: 2026-06-08
Predecessor: `specs/hyde-llm-rerank-experiment.md` (B2 + B3 follow-up)

## 目標 (Goal)

承接前一個 HyDE 實驗的發現 (「瓶頸在 BGE rerank 排序」),量化以下兩個調整在 **完整
question.xlsx 17 題** (gold 在 `answer.xlsx`) 上的命中率影響:

- **B2 — Rerank query enrichment**: `_bge_rerank` 目前傳入 raw `query`。
  改為 `_expand_specs(query)`(已用於向量檢索)或附加 metadata 上下文,
  看 cross-encoder 是否能更精準排序。
- **B3 — Metadata filter strength**: `_hybrid_retrieve` 內 1.3×/0.7× 加分試
  1.5×/0.5×,以及「硬過濾」(只保留 valid_indices) 兩個強度。

## 緣起 (Why)

前實驗發現 7/8 reachable gold 都進到 hybrid retrieve 的 top-50,但 BGE rerank
把它們排太後,top-5 只命中 1 題。如果問題是 reranker 拿到的 query 訊息不足
+ metadata filter 加分太弱,B2/B3 可能直接救起這幾題。

評測集也擴大到全 10 題(含 2 題「無匹配」),不再只看 8 題 reachable。

## 約束 (Constraints)

1. **不改 `rag/engine.py` 的 public API**。實驗碼放 `research/`,直接 import 內部 helper。
2. **不改 corpus / chunks / embeddings / BM25**(這些重建一次要 30+ 分鐘)。
3. **不重跑 `engine.initialize()`** — 一次載入,跑所有變體。
4. **資料源固定為 `question.xlsx` (17 題) + `answer.xlsx` (gold)**。若 gold 在 corpus 內找不到,題目仍進分母並標註「無解」。
5. **每個 variant 對每題只跑一次**,溫度 0,確定性。
6. **產出單一 xlsx** `research/rerank_metadata_tuning_results.xlsx`,gitignored。
7. **若有任一 variant hit@5 顯著 > baseline (≥ +2 題 on n=10)**,才進入「修改 `rag/engine.py`」階段並另寫 contract 測試。否則純研究結論。

## 變體 (Variants)

共 4 個 variant,行列拆解:

|   | rerank query = raw | rerank query = expand+specs |
|---|---|---|
| filter 1.3×/0.7× (current)         | **A. baseline**                | **C. enrich-only**          |
| filter 1.5×/0.5×                   | **B. filter-stronger-only**    | (skip — redundant if C 沒效) |
| filter hard (drop invalid)         | **D. hard-filter + enrich**    |                              |

說明:
- **A**: 完全現狀 (engine.search 的 retrieval+rerank 階段,沒 LLM_select)。
- **B**: 只改 metadata filter 強度。
- **C**: 只改 rerank 看到的 query,加 `_expand_specs(query)` + metadata 摘要
  (例:`"{query}\n[類別] 崁燈\n[瓦數] 15W\n[色溫] 6500K"`)。
- **D**: C 之上把 metadata filter 改成硬過濾。
- 共 4 個 variant × 10 題 = 40 次 rerank。BGE rerank ~1s/題,~40s 總。

## 成功條件 (Success criteria)

1. 對每個 variant 印出 `hit@5`、`hit@1`、`recall@20`(rerank 階段是否進前 20)、`recall@50`(hybrid retrieve 階段)。
2. Per-question 命中表(10 列 × 4 variant 欄)。
3. 若 **任何 variant 比 A 多命中 ≥ 2 題 (hit@5)** → 進「engine 修改」階段:
   - 寫 contract 測試確認沒打壞既有 search() 形狀。
   - 改 `rag/engine.py` 的 `search()` 把 raw query 換成 enriched / 把 1.3/0.7 換成新值。
   - 跑一次 web app 手動 smoke test (`/api/search?q=15W崁燈 6500K`)。
4. 若沒有任何 variant 達標 → 結論「此維度不調」,寫進 spec Results,結案。

## Non-goals

- 不引入 HyDE(已證實無效)。
- 不換 BGE-M3 embedder 或 reranker model。
- 不動 LLM_select 階段(維持現狀)。
- 不評估 latency(rerank 一次 ~1s,差異可忽略)。
- 不擴到 `question.xlsx`(這次只跑 Training.xlsx 10 題)。

## Implementation sketch

```python
# research/rerank_metadata_tuning_experiment.py
from rag import engine
engine.initialize()

def rerank_with_query(query, candidates, top_k, *, rerank_query):
    """Mirror engine._bge_rerank, but with a custom rerank_query."""
    shortlist = candidates[:engine.RERANK_CANDIDATES]
    reranker = engine._get_reranker()
    pairs = [(rerank_query, c["text"][:2000]) for c in shortlist]
    scores = reranker.predict(pairs)
    ordered = sorted(zip(scores, shortlist), key=lambda x: -x[0])
    return [c for _, c in ordered[:top_k]]

def hybrid_with_filter_strength(query, bm25_q, top_k, valid, *, boost, penalty, hard):
    # copy of engine._hybrid_retrieve with (1.3, 0.7) parameterised + optional hard cut
    ...

def metadata_summary(specs):
    # "[類別] 崁燈\n[最大瓦數] 15W"  etc.
    ...

VARIANTS = [
    dict(label="A.baseline",                rerank_q=lambda q: q,
         boost=1.3, penalty=0.7, hard=False),
    dict(label="B.filter-stronger",         rerank_q=lambda q: q,
         boost=1.5, penalty=0.5, hard=False),
    dict(label="C.enrich-rerank",           rerank_q=lambda q, s: f"{engine._expand_specs(q)}\n{metadata_summary(s)}",
         boost=1.3, penalty=0.7, hard=False),
    dict(label="D.hard-filter+enrich",      rerank_q=lambda q, s: f"{engine._expand_specs(q)}\n{metadata_summary(s)}",
         boost=1.3, penalty=0.7, hard=True),
]
```

## 風險 / Fallback

- Hard filter 可能把對的 chunk 過濾掉(metadata 不完整時)→ 預期 D 在「無匹配」兩題會更糟;這正是測試重點。
- Enriched rerank query 變太長 → reranker 自帶 2000 字 truncation,沒問題。
- 若 variant B/C/D 全跟 A 一樣 → 結論「rerank 階段不是真正瓶頸」,下個方向轉向 query expansion / chunking 重做。

## Definition of done

1. Spec 經使用者 `approved` 才開工。
2. `research/rerank_metadata_tuning_experiment.py` 建立並跑完 (~2 分鐘)。
3. 結果寫到此檔尾 `## Results` 段。
4. `research/SUMMARY.md` 增加一行指到本實驗檔。
5. 若有 variant 達標 → 改 `rag/engine.py` 並補 contract 測試 (TBD)。
6. 若沒達標 → 純研究結論,engine 不動。

## Results

### 1. 變體對照 (n=15 scoreable + 2 無匹配)

| Variant                  | hit@5    | hit@1    | @10 rerank | @20 rerank | @50 hybrid |
|--------------------------|----------|----------|------------|------------|------------|
| A. baseline              | 2/15     | 2/15     | 4/15       | 5/15       | **12/15**  |
| B. filter-stronger 1.5/.5| 3/15 (+1)| 3/15     | 4/15       | 6/15       | 12/15      |
| **C. enrich-rerank**     | **5/15 (+3)** | 3/15 | **6/15**   | 6/15       | 12/15      |
| D. hard-filter + enrich  | 4/15 (+3, -1) | 2/15 | 6/15    | 8/15       | 10/15      |

### 2. 各變體 vs baseline 的 diff (hit@5 上)

- **B.filter-stronger** +Q5 (IP66 嚴格 IP 過濾受惠)
- **C.enrich-rerank** +Q2, Q11, Q17 — 沒掉任何題
- **D.hard+enrich** 同 C 撈回 3 題,但 **掉了 Q15 (L4140R5)** — 該 chunk 的 metadata 不完整被硬過濾誤殺,印證 spec 內的風險

### 3. 關鍵觀察

**瓶頸確認在 rerank query**:`@50 hybrid` 命中 12/15 全部一致 — A/B/C 三變體 retrieve 階段
撈到的東西一模一樣,差別純粹在 cross-encoder 看到的 query。把 raw query
換成 `expand_specs(query) + metadata 摘要`,@10 從 4 拉到 6,top-5 從 2 拉到 5。

**Metadata filter 強度影響不大**:B (1.5/0.5) 比 A 只多 1 題,且 D 硬過濾還會誤殺
metadata 不完整的好 chunk。soft boost 1.3/0.7 已經夠用,不要往上加。

**剩下 7 題沒救**(@50 hybrid 也沒中):
- Q1, Q3, Q4, Q6, Q8, Q10 — gold 不在 corpus 內或 BM25/向量都打不到
- Q14 — gold (LED-2441R1, D-T810DR9) 部分不存在(同 Training.xlsx 結論)

### 4. 結論:採用 C.enrich-rerank

**Engine 修改方案** (最小 diff):

```python
# rag/engine.py:684 — search()
- reranked = _bge_rerank(query, candidates, top_k=LLM_SELECT_CANDIDATES)
+ rerank_query = _build_rerank_query(query, specs)
+ reranked = _bge_rerank(rerank_query, candidates, top_k=LLM_SELECT_CANDIDATES)
```

新增私有 helper `_build_rerank_query(query, specs)` = `expand_specs(query) +
metadata_summary(specs)`,後者把 `_decompose_query` 的 dict 渲染成
`[類別] xxx / [最大瓦數] N W / [色溫] N K / [最小光通量] N lm / [IP] IPN` 多行字串。

### 5. 後續工作 (out of this spec)

- **B1**(從前實驗承接):清理 question.xlsx 的 Q1/Q3/Q4/Q6/Q8/Q10/Q14 — 多數 gold 不在 corpus。
- **B4**:Q1/Q6/Q8 在 hybrid 階段就沒中,需 BM25 加入 model-name 前綴或調 BM25 token。
- 不採用 D 硬過濾;不採用 B 強度提升。

### Artifacts

- 腳本:`research/rerank_metadata_tuning_experiment.py`
- 結果 xlsx:`research/rerank_metadata_tuning_results.xlsx` (gitignored)
- Console log:`/tmp/rerank_exp.log` (machine-local)
