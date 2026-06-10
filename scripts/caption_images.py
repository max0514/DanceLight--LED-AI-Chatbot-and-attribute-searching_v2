"""One-shot: caption every page image with minicpm-v via Ollama, persist to
img_descriptions_cache.json. Idempotent; restarts from cache. Independent of
research/pipeline.py so we can rebuild captions without running the benchmark.

Usage (from repo root):
    python3 scripts/caption_images.py
"""
from __future__ import annotations

import base64
import json
import os
import signal
import sys
import time
from pathlib import Path

import ollama

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)

ODL_JSON = "./output_opendataloader/2026舞光LED22st(單頁).json"
ODL_DIR = "./output_opendataloader"
IMG_CACHE = "./img_descriptions_cache.json"
VISION_MODEL = "minicpm-v"
TIMEOUT_S = 60
MIN_IMG_SIZE = 1000

VISION_PROMPT = (
    "請用繁體中文簡要描述這張燈具產品圖片的重要資訊，特別注意：\n"
    "1. 圖片上標註的角度數字（如旋轉角度、光束角）\n"
    "2. 產品型號文字\n"
    "3. 尺寸標註\n"
    "4. 產品的外觀特徵（形狀、顏色、安裝方式）\n"
    "請只描述圖片中看到的內容。用50-100字簡述。"
)


def _load_cache() -> dict:
    if os.path.exists(IMG_CACHE):
        with open(IMG_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict) -> None:
    with open(IMG_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def main() -> None:
    with open(ODL_JSON, "r") as f:
        odl = json.load(f)

    paths: list[str] = []
    seen: set[str] = set()
    for kid in odl["kids"]:
        if kid.get("type") != "image":
            continue
        src = kid.get("source") or ""
        if not src:
            continue
        fp = os.path.join(ODL_DIR, src)
        if fp in seen:
            continue
        seen.add(fp)
        if not os.path.exists(fp):
            continue
        if os.path.getsize(fp) < MIN_IMG_SIZE:
            continue
        paths.append(fp)

    cache = _load_cache()
    todo = [fp for fp in paths if fp not in cache]
    print(f"[caption] total {len(paths)}, cached {len(paths)-len(todo)}, todo {len(todo)}")

    n = 0
    t_start = time.time()
    for fp in todo:
        n += 1
        try:
            with open(fp, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            def _to(signum, frame):
                raise TimeoutError()

            old = signal.signal(signal.SIGALRM, _to)
            signal.alarm(TIMEOUT_S)
            t0 = time.time()
            resp = ollama.chat(model=VISION_MODEL, messages=[{
                "role": "user", "content": VISION_PROMPT, "images": [img_b64],
            }])
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old)
            desc = resp["message"]["content"].strip()
            dt = time.time() - t0
        except TimeoutError:
            signal.alarm(0)
            desc = "(超時)"
            dt = TIMEOUT_S
        except Exception as e:
            signal.alarm(0)
            desc = f"(失敗: {type(e).__name__})"
            dt = 0.0

        cache[fp] = desc
        if n % 20 == 0:
            _save_cache(cache)
            elapsed = time.time() - t_start
            eta = elapsed / n * (len(todo) - n) / 60
            print(f"[caption] {n}/{len(todo)} done | last {dt:4.1f}s | eta {eta:5.1f}min")

    _save_cache(cache)
    print(f"[caption] done — wrote {n} new entries to {IMG_CACHE}")


if __name__ == "__main__":
    main()
