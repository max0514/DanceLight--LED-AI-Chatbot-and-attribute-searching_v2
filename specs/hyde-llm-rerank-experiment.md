# Spec: HyDE / LLM-rerank Ablation on Training.xlsx

Status: **COMPLETED — 2026-06-08**
Owner: Claude Code (research)
Created: 2026-06-08
Outcome: 結論 = 「都不要加」(詳見 Results)

## 目標 (Goal)
量化「加 HyDE」與「加 HyDE + LLM reranker」相對於 baseline 在 `Training.xlsx`
(10 題標案規格) 的命中率差異,選出對舞光 LED 型錄 RAG 最有幫助的組合。

## 緣起 (Why)
使用者看到一張投影片裡的企業級 RAG 架構 (HyDE rewrite + LLM reranker),想知道把
這兩段加進現行 pipeline 是否值得。

**重要 baseline 釐清** (我先前比較有誤):
- `rag/engine.py` (web app) **已經有 GPT-4o picks 5** 這段 — 也就是 #2 LLM reranker
  其實已經內建。
- `rag/engine.py` **沒有 HyDE** — 向量查詢用 `expand_specs(query)` (regex spec 展開)
  作為輸入,不是 LLM 改寫的 hypothetical passage。

因此本實驗的三個變體實際上是:

| Variant | Vector query | After hybrid | After BGE rerank | 對應使用者問題的 |
|---|---|---|---|---|
| **A. Baseline** | `expand_specs(query)` | BGE rerank top-20 | **(無)** 取 top-5 | 「都不加」 |
| **B. +HyDE** | `expand_specs(query) + hyde_passage` | BGE rerank top-20 | **(無)** 取 top-5 | 「+ #1」 |
| **C. +HyDE +LLM-rerank** | `expand_specs(query) + hyde_passage` | BGE rerank top-20 | **GPT-4o picks 5** | 「+ #1+#2」 |

注意 C 就是現行 web app 的 search() **再加上 HyDE**;B 是純向量改進、不靠 LLM 決選。

## 約束 (Constraints)
1. **不改 `rag/engine.py` 的 public API** — 實驗碼放 `research/`,匯入 engine 的
   內部 helper (`_hybrid_retrieve` / `_bge_rerank` / `_llm_select` 等) 跑三遍。
2. **資料源固定為 `Training.xlsx`** (10 列,2 欄: 詢問問題 / 期望回答)。
3. **不重新載入 BGE-M3 / reranker** — 三個 variant 共用一次 `engine.initialize()`。
4. **HyDE 用 gpt-4o-mini** (便宜、快、JSON-mode 可控)。每題 ~$0.0002,10 題 ~$0.002。
5. **LLM reranker 沿用 `engine._llm_select`** 的 gpt-4o,不另起爐灶。
6. **每個 variant 對每題只跑一次**,溫度 0,結果可重現。
7. **產出單一 xlsx** (`research/hyde_llm_rerank_results.xlsx`) 含三個變體的 top-5 與
   命中欄,gitignored (output 不入 repo)。

## 成功條件 (Success criteria)
1. 對每個 variant,印出:
   - `hit@5` (gold 型號出現在 top-5 任一筆): 主指標
   - `hit@1` (gold 型號出現在 top-1): 嚴格指標
   - 對 `無匹配產品` 兩題,記錄「是否仍 confidently 回覆」(只 C 變體可正確回 nothing)
2. 印出對照表:
   ```
                  hit@5   hit@1   no-match handled
   A. baseline    X/8     X/8     (n/a)
   B. +HyDE       X/8     X/8     (n/a)
   C. +HyDE+LLM   X/8     X/8     X/2
   ```
3. 對每題印出三個變體分別命中/未中,方便 diff。
4. 整個實驗 < 5 分鐘跑完 (10 題 × 3 variant,主要時間花在 BGE rerank + LLM call)。
5. Spec / 結果 / 結論寫進 `specs/hyde-llm-rerank-experiment.md` 的「Results」段落
   (寫在這份檔尾)。

## Non-goals
- 不調 BGE reranker / hybrid weights。
- 不換 embedder (固定 BGE-M3)。
- 不跑 `question.xlsx` 17 題。Training.xlsx 是使用者選的測試集。
- 不引入 RRF / Cohere rerank。
- 不評估 latency (本實驗只看命中率)。

## Open questions
1. **HyDE 該用 gpt-4o-mini 還是本機 Ollama?** → 建議 gpt-4o-mini:
   (a) 成本可忽略,(b) 結果可重現,(c) 不依賴本機 Ollama 是否在跑。
2. **HyDE 拼接策略?** → 建議 `f"{expanded_query}\n\n[HyDE]\n{hyde_passage}"`,
   保留原 query 訊號避免被 HyDE 帶歪。
3. **Top-K?** → 主指標用 hit@5 (與 web app 一致)。

## Implementation sketch
```
research/
  hyde_llm_rerank_experiment.py    # 新檔
```

腳本概要:
```python
from rag import engine
engine.initialize()

def hyde_generate(query: str) -> str:
    # gpt-4o-mini, JSON mode, 100-150 字 hypothetical 型錄條目
    ...

def retrieve_variant(query: str, *, use_hyde: bool, use_llm_rerank: bool, top_k: int = 5):
    # 沿用 engine 內部 helpers,參數化 vector query 與是否走 _llm_select
    specs = engine._decompose_query(query)
    valid = engine._metadata_filter(specs, engine._state["chunk_metas"])
    expanded = engine._expand_specs(query)
    bm25_q = engine._add_synonyms(query)
    vec_q = expanded + (f"\n\n[HyDE]\n{hyde_generate(query)}" if use_hyde else "")
    cands = engine._hybrid_retrieve(vec_q, bm25_q, engine.RETRIEVE_K, valid)
    reranked = engine._bge_rerank(query, cands, top_k=engine.LLM_SELECT_CANDIDATES)
    if use_llm_rerank:
        return engine._llm_select(query, reranked, top_k=top_k)
    return reranked[:top_k]

# Score each variant against Training.xlsx
# 用 research/pipeline.py 已存在的 extract_model_numbers 邏輯做命中判斷
```

## 風險 / Fallback
- HyDE prompt 不夠精準會引入雜訊 → 用 prompt 嚴格規定「只生成 1 段型錄體型文字,
  100 字以內」, 並輸出原 query 為對照。
- LLM reranker 失敗 → `engine._llm_select` 自帶 fallback (沿用 BGE rerank 序)。
- 若實驗結果 B 與 A 接近(差距 ≤ 1 題,n=8 太小),結論為「資料量不足以判斷」,
  建議擴大到 `question.xlsx` 17 題再跑一次。

## Definition of done
1. Spec 經使用者 `approved` 才開工。
2. `research/hyde_llm_rerank_experiment.py` 建立並跑完。
3. 結果寫到此檔尾 `## Results` 段。
4. `research/SUMMARY.md` 增加一行指到本實驗檔。
5. 不需要 contract 測試 (一次性研究腳本,非長期維護的契約)。
6. Commit 訊息: `Add HyDE / LLM-rerank ablation experiment on Training.xlsx`。

## Results

### 1. 變體對照 (hit@5)

| Variant                  | hit@5 (有 gold) | hit@1 | 無匹配拒答 |
|--------------------------|-----------------|-------|-----------|
| A. baseline              | **1/8**         | 1/8   | n/a       |
| B. +HyDE-mini            | 1/8             | 1/8   | n/a       |
| B. +HyDE-local (qwen3.6) | 1/8             | 1/8   | n/a       |
| C. +HyDE-mini+LLM        | **2/8**         | 1/8   | 0/2       |
| C. +HyDE-local+LLM       | 2/8             | 1/8   | 0/2       |

唯一差異:C 變體 (有 LLM rerank) 多撈到 Q1 (E-FLCS50D)。HyDE provider 換成本地完全無差別。

### 2. 真正的瓶頸不在 HyDE / LLM-rerank,在 BGE rerank 排序

我額外算了 baseline 在不同深度的 recall:

| 深度 | 8 題裡命中 | 備註 |
|---|---|---|
| hit@5  | 1/8 | 主指標 |
| hit@10 | 2/8 | +Q4 |
| hit@20 | 3/8 | +Q1 |
| hit@50 | **7/8** | Q3 corpus 內找不到型號,無解 |

**Gold 幾乎都在 BGE rerank top-50 內**,但 reranker 把它們排得太後。HyDE 無法影響這個排序;
LLM-rerank 也只挑回 1 題 (Q1)。瓶頸 = **BGE-reranker 在中文型錄場景對「規格條件」不夠敏感**。

### 3. 評測集本身有問題

Training.xlsx 10 題裡,**3 個 gold 型號根本不在 PDF 內**:
- Q3 `D-21DOP25NR2` — 整本型錄找不到
- Q7 `D-T810DR9` — 找不到 (Q7 另一個 `LED-2441R1` 在)
- Q8 `DSTMS` — 找不到 (Q8 另一個 `L4140R5` 在)

這些題不論怎麼改 pipeline 都不可能命中。Eval set 需要先清理。

### 4. HyDE 觀察

- **gpt-4o-mini 會幻覺型號** — 例:給 Q1 (要 E-FLCS50D),HyDE 生出 `E-FLCT50D`、`E-FLCS40D`、
  `E-FLCS25D-4000`,格式對但全是編造的。這些假型號進向量檢索沒帶來正面訊號,但也沒明顯
  害到 (因為向量是語意相似,不是字串匹配)。
- **qwen3.6:latest 對部分 query 輸出空字串** — 推測是內部 `<think>` 段把全部內容包住,
  strip 完就空了。num_predict=300 可能也不夠。本地模型在這個任務上不穩。

### 5. LLM rerank 觀察

C 變體唯一加分的是 Q1。Q4 的 gold (D-PD25N-EGR1) **在 BGE rerank top-20 第 6 名**,LLM 看到也沒挑,
顯示 gpt-4o 在「20 個都很相似的崁燈條目」中沒有強區辨力。「無匹配」兩題 LLM 仍硬吐 5 個,
prompt 寫死「即使規格不完全吻合,也要列出 5 個」是主因 — 對搜尋體驗合理,但無法用來自動拒答。

### 結論與下一步

**1. 不採用 HyDE。** 兩個 provider 都沒帶來可量測的好處,還引入 API 成本/本地不穩/幻覺型號風險。

**2. LLM rerank (Variant C 的後半段) 已經在 web app 用了** — 維持現狀,不再疊 HyDE。

**3. 真正該做的事**(若有時間,另開 spec):
- **B1**:把 Training.xlsx 缺失的 3 個 gold 補進 PDF 來源,或改 eval set。
- **B2**:把 BGE rerank 的 query 從 raw `query` 換成 `expand_specs(query)` 或加 metadata 上下文 —
  目前 reranker 看到的 query 太短、太通用,難排出精確規格匹配。
- **B3**:metadata filter 1.3×/0.7× 加分試 1.5×/0.5× 或硬過濾,讓符合「色溫 + 瓦數 + IP」的
  chunk 直接搶到 top-N。
- **B4**:研究 query-side 加 chunk model 名稱前綴提示(像 BM25 一樣強化字串訊號)。

### Artifacts

- 腳本:`research/hyde_llm_rerank_experiment.py`
- HyDE 快取:`research/.hyde_cache.json` (gitignored)
- 詳細結果 xlsx:`research/hyde_llm_rerank_results.xlsx` (gitignored)
- 完整 console log:`/tmp/hyde_exp.log` (machine-local)

