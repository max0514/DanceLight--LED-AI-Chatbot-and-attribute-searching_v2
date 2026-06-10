# Spec: dual-edition selector (21st + 22nd) in the web app

**Status**: draft
**Author**: Claude  **Date**: 2026-06-10
**Affected modules**: rag / web

---

## 目標 (Goal)

使用者在 web app 上可以切換 21 版 / 22 版型錄,每個查詢用對應版本的 corpus 回答。
22 版是 default;切換不需要重啟 server,UI 只是一個下拉/Toggle。後端
`/api/search` 接受 `edition` 欄位,engine 同時掛載兩版 corpus 在記憶體裡,
路由零延遲。

---

## 約束 (Constraints)

- [ ] `rag/CONTRACT.md` — `search(query, top_k=5)` 必須仍可呼叫(向後相容),
      新增 `edition` 為 keyword-only optional 參數,預設 `"22nd"`
- [ ] `web/CONTRACT.md` — `/api/search` 回傳 JSON shape 不變,只增加可選輸入欄位 `edition`
- [ ] Embedding cache invariant 仍為 chunk_text 的 md5 key
- [ ] `annotations_cache.json` 不分版(md5-keyed → 各版條目並存,不互相覆蓋)
- [ ] `img_descriptions_cache.json` 不分版(file path key 已含版本目錄,天然分隔)
- [ ] BGE-M3 reranker / embedder 只各載入一次(共用),不複製 GPU 記憶體
- [ ] 啟動成本可上升,但每查詢延遲不變(p95 < 2s 不變)
- [ ] tunnel `dancelight-rag-nccu.loca.lt` URL 不變
- [ ] cron supervisor 不改 (`/usr/local/sbin/dancelight-web-ensure.sh`)

---

## 成功條件 (Success criteria)

- [ ] `rag.engine.search("15W崁燈 6500K", edition="21st")` 回 5 個 21 版產品
- [ ] `rag.engine.search("15W崁燈 6500K", edition="22nd")` 回 5 個 22 版產品 (內含可能新型號)
- [ ] `rag.engine.search("15W崁燈 6500K")` — 沒帶 edition,默認 22 版,行為等同 today
- [ ] `POST /api/search {"query":"...","edition":"21st"}` 與 `{"edition":"22nd"}` 各別回對應版本
- [ ] `POST /api/search {"query":"..."}` 不帶 edition → 默認 22 版
- [ ] Web UI 顯示版本選擇器(下拉或 toggle),切換後下一次查詢用新版
- [ ] `rag.engine.initialize()` 同時掛載兩版,啟動時間從目前 ~30s 增加到 ~35s (可接受)
- [ ] 兩版的 `chunk_embeddings_<ed>.npy` 都存在且 row count 對得上 chunks 數

---

## Non-goals

- 同時查詢兩版做 cross-edition 比對(只切換,不合併)
- 把 21 版的 ColBERT cache 也同步重建 (experiment.engine 不在 web app path)
- 重新 author `question.xlsx` / `answer.xlsx` 對齊任一版
- 21 版的 image captions 補滿 100%(目前 202/388 = 52% 已可運作,可選擇追加)

---

## 影響檔案 (Files touched)

| File | Change |
|---|---|
| `rag/engine.py` | 新增 `CORPUS` dict 內含兩版設定,`_states[edition]` 取代 `_state`;`initialize(edition=None)` (None → 兩版都載入);`search(..., *, edition="22nd")` |
| `rag/CONTRACT.md` | 標註 `edition` 可選參數;`PDF_PATH` 改為 `CORPUS` mapping |
| `web/app.py` | `/api/search` 接收 `edition` 欄位;Flask route 邊讀邊驗證 enum |
| `web/CONTRACT.md` | API 加入 optional `edition` 欄位文件 |
| `web/templates/index.html` | UI 加版本選擇器(預設 22 版),前端帶 edition 到 fetch |
| `bge_m3_embeddings/chunk_embeddings_21st.npy` | 新建 (BGE-M3 重跑 21 版 chunks) |
| `bge_m3_embeddings/chunk_embeddings_22nd.npy` | rename 自既有 `chunk_embeddings.npy` |
| `bge_m3_embeddings/chunk_embeddings.npy` | 刪除(被 _22nd 取代);若需 backward compat 留 symlink |

---

## 設計 (Design)

### 1. Engine 雙 corpus

```python
CORPUS = {
    "21st": {
        "pdf": "./2025舞光LED21st(單頁水印可搜尋).pdf",
        "odl_json": "./output_opendataloader/2025舞光LED21st(單頁水印可搜尋).json",
        "embed_cache": "./bge_m3_embeddings/chunk_embeddings_21st.npy",
    },
    "22nd": {
        "pdf": "./2026舞光LED22st(單頁).pdf",
        "odl_json": "./output_opendataloader/2026舞光LED22st(單頁).json",
        "embed_cache": "./bge_m3_embeddings/chunk_embeddings_22nd.npy",
    },
}
_states: dict[str, dict] = {}     # keyed by edition

def initialize(edition: str | None = None) -> None:
    """Load given edition (or all when None). Idempotent per edition."""
    targets = list(CORPUS) if edition is None else [edition]
    for ed in targets:
        if ed in _states:
            continue
        cfg = CORPUS[ed]
        # ... existing chunking/embedding logic, but parameterised by cfg
        _states[ed] = {...}

def search(query: str, top_k: int = 5, *, edition: str = "22nd") -> list[dict]:
    if edition not in CORPUS:
        raise ValueError(f"unknown edition {edition!r}")
    if edition not in _states:
        initialize(edition)
    state = _states[edition]
    # ... use `state` everywhere instead of module-level `_state`
```

PDF page rendering (used by `/page_image/<page>`) needs to know which PDF.
Either route through `_states[edition]["pdf_doc"]` or open per request.

### 2. Web API + UI

`/api/search` 接 JSON `{query, top_k?, edition?}`. Validate `edition in {"21st", "22nd"}`,
fallback `"22nd"` 若缺。傳給 `engine.search(..., edition=edition)`.

`/page_image/<page>` 也需要知道是哪一版。最簡單做法是改成
`/page_image/<edition>/<page>`,前端 fetch 時帶上目前選的版本;
舊路由 `/page_image/<page>` 保留為 22 版的 alias (backward compat)。

UI:
```html
<div class="edition-selector">
  <label><input type="radio" name="ed" value="22nd" checked> 22 版 (2026, 420頁)</label>
  <label><input type="radio" name="ed" value="21st"> 21 版 (2025, 388頁)</label>
</div>
```

JS state: 讀 `document.querySelector('input[name=ed]:checked').value`,
fetch /api/search 時放進 body,fetch /page_image 時放進 URL。

### 3. Cache rebuild

22 版 embeddings 目前在 `chunk_embeddings.npy`;rename 為 `chunk_embeddings_22nd.npy`.
21 版需重跑 BGE-M3 一次 (~6s on GPU,等同 22 版的數字),寫到
`chunk_embeddings_21st.npy`.

`initialize(edition)` 觸發 auto-build 的邏輯仍然有效 (已對齊 CONTRACT)。

---

## 風險與緩解

- **風險**: 雙 corpus 把記憶體 footprint 翻倍。
  **緩解**: 估計 ~50 MB 增量 (chunks + BM25 + embeddings 都小;reranker/embedder
  共用)。實測無壓力。

- **風險**: 21 版的 image captions 只有 52% 覆蓋率,部分 chunks 缺圖文。
  **緩解**: 21 版功能仍可用,只是部分頁面 chunk text 不含圖描述。可選擇後補。

- **風險**: 21 版的 `question.xlsx` 對 21 版 corpus 仍有效,但 9/15 的 hit@5 結果
  本來就是基於那組 eval set + 21 版的。切回 21 版時 retrieval 應該與舊系統一致。
  **緩解**: 用 `scripts/smoke_test.py` 雙版本驗證。

- **風險**: cron supervisor 啟動 webapp 時,雙 corpus 載入需要更久 (~35s 首次),
  期間 :8000 沒人接,supervisor 可能會再 spawn 一個 → 雙啟動。
  **緩解**: supervisor 用 `ss -tln :8000` 與 `pgrep -f "python3 -m web.app"` 雙重檢查
  (已現存 logic), 啟動中 pgrep 已能擋住。
