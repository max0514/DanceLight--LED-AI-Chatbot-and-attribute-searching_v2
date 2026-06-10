# Spec: upgrade source corpus from 21st to 22nd edition

**Status**: draft
**Author**: Claude  **Date**: 2026-06-10
**Affected modules**: rag / web / research / experiment

---

## 目標 (Goal)

舞光於 2026-04 發行第 22 版 LED 型錄 PDF (`2026舞光LED22st(單頁).pdf`,
381 MB, 420 頁,比 21 版多 32 頁),內含當年度新品 + 部分舊款下架。
本 spec 把 RAG 系統的 source corpus 從 21 版整批換成 22 版。換完後
web app 的搜尋結果反映最新型錄,research/experiment 的 benchmark 也跑
22 版的內容。

---

## 約束 (Constraints)

- [ ] `rag/CONTRACT.md` — search() 介面 (signature, return shape) 不變,
      只有 underlying corpus 變
- [ ] `web/CONTRACT.md` — `/api/search` JSON shape 不變,page_image_url
      指向新版的 page images
- [ ] 不改動 chunking 邏輯、不換 embedding model、不換 reranker
- [ ] Embedding cache invariant 仍為 chunk_text 的 md5 key —
      新版 chunk 內容跟舊版 md5 不同 → 全部 miss,觸發重建,符合預期
- [ ] 21 版的舊 caches 與舊 PDF 不删除,只保留在 NAS 備份位置;repo
      symlink 與 code 一律指向 22 版
- [ ] Public URL (https://dancelight-rag-nccu.loca.lt) 在切換過程中仍可服務舊內容,
      直到所有 cache rebuild 完成才切到新內容 — 不能有破窗期

---

## 成功條件 (Success criteria)

- [ ] `rag/engine.py` PDF_PATH / ODL_JSON 指向 22nd-edition 檔
- [ ] `research/pipeline.py` 三個路徑常數同上
- [ ] `/var/lib/jenkins/dancelight_data/output_opendataloader/2026舞光LED22st(單頁).json` 存在
- [ ] `bge_m3_embeddings/chunk_embeddings.npy` rows = 22nd-edition chunks 數
- [ ] `annotations_cache.json` keys cover 100% of 22nd-edition chunks
- [ ] `img_descriptions_cache.json` keys cover 22nd-edition page images
- [ ] Smoke test: `python -c "from rag import engine; engine.initialize();
      r=engine.search('15W崁燈 6500K', top_k=3); print(len(r))"` 回 3 結果
- [ ] Web app 端到端: 重啟後一個查詢能正常回傳 5 個 cards
- [ ] `experiment/engine.py` ColBERT cache (`chunk_tokens.npz`) 重建後
      `experiment/bench.py` 至少能跑完 (accuracy 數字會跟之前不同,屬預期)

---

## Non-goals

- 重新 author `question.xlsx` / `answer.xlsx` (eval set 對齊 22 版需要舞光
  端的 ground truth) — 留到後續 spec
- 把 21 版的 caches 從 NAS 備份刪掉
- 升級 chunking、embedding model 或任何演算法 — 純粹是換 source PDF
- 把先前 strict30/pure 9/15 hit@5 的成果 graduate 到 rag/engine.py —
  那個結果對齊的是 21 版 corpus,要先有 22 版的新 baseline 才談 graduate
- 改 README / SUMMARY 之外的文件結構

---

## 影響檔案 (Files touched)

| File | Change |
|---|---|
| `rag/engine.py` | `PDF_PATH`, `ODL_JSON` 路徑常數 |
| `research/pipeline.py` | `PDF_PATH`, `ODL_JSON`, `ODL_IMG_DIR` 常數 + 開頭註解的 388→420 頁 |
| `experiment/engine.py` | 不改 code;但 `chunk_tokens.npz` 會失效需重建 |
| `.gitignore` | `2025舞光LED21st*.pdf` → `2025舞光LED21st*.pdf` + `2026舞光LED22st*.pdf` (二者都不入版本控制) |
| `2025舞光LED21st(單頁水印可搜尋).pdf` (symlink) | 移除,改建 `2026舞光LED22st(單頁).pdf` symlink |
| `README.md` | 388-page → 420-page;檔名更新 |
| `rag/SUMMARY.md` / `research/SUMMARY.md` | 提到的版次更新 |
| `MEMORY.md` 內相關 entry | 無需改 (沒有指向 PDF 版次的 memory) |

---

## 重建步驟 (執行順序)

1. opendataloader-pdf 跑 22 版 → 產生 JSON + page images (執行中)
2. 改 code 路徑常數 + 重新 symlink (純文字編輯,5 分鐘)
3. 跑一個短 Python script:
   - `from rag import engine; engine.initialize()` — 會觸發:
     - chunking 22nd PDF
     - annotation_cache miss → 全頁觸發 gpt-4o-mini (約 30 min,$0.55)
     - img_descriptions_cache miss → minicpm-v 對新 page images 跑 (約 20 min)
     - BGE-M3 dense embeddings 重建 → 寫 `chunk_embeddings.npy` (數分鐘)
4. 跑 `from experiment import engine as xe; xe._build_or_load_colbert()` — ColBERT
   tokens cache 重建 (數小時 GPU)
5. Smoke test + restart web app (kill 既有 process,重啟 `python3 -m web.app`)

每一步都可以單獨重跑,只要 cache 檔不存在或 chunk md5 mismatch 就會自動觸發。

---

## 風險與緩解

- **風險**: rebuild 過程中 web app 仍須對外服務。
  **緩解**: 不動既有 server process,新的 caches 寫到新檔名前不影響舊路徑;
  最後一步才 restart server。

- **風險**: 22 版的 chunking 出問題 (例如新版改了排版,regex 抓不到 model token)。
  **緩解**: opendataloader JSON 跑完後先抽樣 5 頁人工確認 type/bounding box 結構
  跟 21 版一致再進行下一步。

- **風險**: OpenAI quota / API 失敗造成 annotation rebuild 中斷。
  **緩解**: cache 是 incremental — 重跑同 chunks 會從上次斷點繼續。

- **風險**: 21 版 question.xlsx 對 22 版 corpus 沒有意義,benchmark 數字看不懂。
  **緩解**: 在 README + summary 註明 baseline 已重置;experiment.bench 跑完
  後在 spec 寫 Results section 標明 "22 版 baseline (eval set 未更新)"。
