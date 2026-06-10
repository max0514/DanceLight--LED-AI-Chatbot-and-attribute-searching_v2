"""Post-swap smoke test: chunks load, search runs end-to-end, returns 5 docs.

Usage (from repo root):
    python3 scripts/smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(".env")

from rag import engine

t0 = time.time()
print("[smoke] initialize() — all editions ...")
engine.initialize()
for ed, st in engine._states.items():
    print(f"[smoke]   {ed}: {len(st['chunks'])} chunks, "
          f"embeds {st['chunk_embeddings'].shape}")
print(f"[smoke] ready in {time.time()-t0:.1f}s")

QUERIES = [
    "15W崁燈 6500K",
    "LED 投射燈 ≦50W 6000K IP65",
    "T-BAR LED 平板燈 ≦35W 4000K",
    "LED 步道燈 9W 220V",
]

for ed in engine._states:
    print(f"\n=== edition: {ed} ===")
    for q in QUERIES:
        t = time.time()
        docs = engine.search(q, top_k=5, edition=ed)
        print(f"[smoke] '{q}' ({time.time()-t:.1f}s):")
        for d in docs:
            models = (d.get("models") or "")[:30]
            page = d.get("page", "?")
            print(f"  {d.get('rank_label','?'):<8s} p.{page} {models} | {d.get('name','')[:30]}")
        if not docs:
            print("  (no results)")
            sys.exit(1)

print("\n[smoke] OK")
