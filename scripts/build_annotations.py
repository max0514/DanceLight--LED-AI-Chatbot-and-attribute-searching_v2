"""Rebuild annotations_cache.json against the current PDF / chunking.

Calls rag.engine.initialize() to get the same chunk set the engine will use at
runtime (md5 keys then match). For each chunk not already in the cache,
generates a structured YAML annotation via gpt-4o-mini.

Usage (from repo root):
    OPENAI_API_KEY=... python3 scripts/build_annotations.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

load_dotenv(".env")
from rag import engine  # noqa: E402

ANNOTATIONS_CACHE_PATH = "./annotations_cache.json"
ANNOTATION_LLM = "gpt-4o-mini"
ANNOTATION_MAX_CHARS = 1500
SYN_XLSX_PATH = "./category_synonyms.xlsx"

ANNOTATION_PROMPT_TMPL = """你是舞光 (Dancelight) LED 型錄資料標註員。任務：為下方「型錄節錄」產生一段結構化 YAML 註解，
讓後續的 RAG 系統 (BM25 + BGE-M3 + reranker + LLM) 更準確地把客戶問題對映到正確產品。

【嚴格規則】
1. 只能根據型錄節錄中**實際出現的文字**填寫，不得自行推論或新增規格。
2. 找不到的欄位填「未提供」，禁止留白或編造。
3. 產品類別 (canonical_category) 從下列正式名稱選 1 個；找不到對應則填「其他」：
   {canonical_list}
4. aliases_present 只能列出**節錄文字確實出現過**的別名 (從下表挑)：
   {alias_table}
5. 全部用繁體中文；YAML 區塊整體 ≤ 200 字。

【輸出格式】(只輸出純 YAML，不要任何前後說明)
canonical_category: <步驟 3 的正式名稱>
model_codes: [<型號1>, <型號2>, ...]
key_specs:
  wattage: <如 "30W" 或 "未提供">
  color_temp: <如 "3000K/4000K/6500K" 或 "未提供">
  lumens: <如 "3000lm" 或 "未提供">
  ip_rating: <如 "IP65" 或 "未提供">
  voltage: <如 "AC100-240V" 或 "未提供">
aliases_present: [<節錄出現的別名>]
use_cases: [<節錄明確提到的應用場景，最多 3>]
one_line_summary: <≤40 字一句話描述「這是哪一類、哪個型號、給誰用」>

【型錄節錄】
{chunk_text}

【註解輸出】
"""


def _hash_chunk(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def _load_synonyms() -> tuple[list[str], str]:
    df = pd.read_excel(SYN_XLSX_PATH, sheet_name=0)
    canonicals: list[str] = []
    alias_rows: list[str] = []
    for _, row in df.iterrows():
        canon = str(row.iloc[0]).strip()
        if not canon or canon.startswith("填寫說明") or canon == "nan":
            continue
        canonicals.append(canon)
        aliases = [str(x).strip() for x in row.iloc[1:] if str(x).strip() and str(x) != "nan"]
        if aliases:
            alias_rows.append(f"  {canon}: {', '.join(aliases)}")
        else:
            alias_rows.append(f"  {canon}: (無別名)")
    return canonicals, "\n".join(alias_rows)


def main() -> None:
    print("[anno] loading chunks via rag.engine.initialize() (all editions) ...")
    engine.initialize()
    chunks: list[str] = []
    seen_md5: set[str] = set()
    for ed, st in engine._states.items():
        for c in st["chunks"]:
            h = _hash_chunk(c)
            if h in seen_md5:
                continue
            seen_md5.add(h)
            chunks.append(c)
        print(f"[anno]   {ed}: {len(st['chunks'])} chunks")
    print(f"[anno] union: {len(chunks)} unique chunks across all editions")

    canonical_list, alias_table_str = _load_synonyms()
    print(f"[anno] {len(canonical_list)} canonical categories loaded")

    if os.path.exists(ANNOTATIONS_CACHE_PATH):
        with open(ANNOTATIONS_CACHE_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {}

    todo = [(i, _hash_chunk(c), c) for i, c in enumerate(chunks)
            if _hash_chunk(c) not in db]
    print(f"[anno] cached {len(db)}, todo {len(todo)} / {len(chunks)}")
    if not todo:
        print("[anno] nothing to do")
        return

    client = openai.OpenAI()
    t0 = time.time()
    for k, (idx, h, txt) in enumerate(todo, 1):
        prompt = ANNOTATION_PROMPT_TMPL.format(
            canonical_list=", ".join(canonical_list),
            alias_table=alias_table_str,
            chunk_text=txt[:ANNOTATION_MAX_CHARS],
        )
        try:
            resp = client.chat.completions.create(
                model=ANNOTATION_LLM,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0,
            )
            db[h] = resp.choices[0].message.content.strip()
        except Exception as e:
            db[h] = f"# annotation failed: {type(e).__name__}: {e}"
        if k % 25 == 0 or k == len(todo):
            elapsed = time.time() - t0
            print(f"[anno] {k}/{len(todo)} ({elapsed:.0f}s, avg {elapsed/k:.2f}s/chunk)")
            with open(ANNOTATIONS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False, indent=2)

    with open(ANNOTATIONS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"[anno] done — wrote {ANNOTATIONS_CACHE_PATH}")


if __name__ == "__main__":
    main()
