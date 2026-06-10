"""Run experiment.engine.search against question.xlsx + answer.xlsx.

Goal: hit@5 ≥ 11/15 (70%).

Usage:
    python3 -m experiment.bench                     # default config
    python3 -m experiment.bench --variant enrich    # named variants (see VARIANTS)
    python3 -m experiment.bench --cfg '{"enrich_rerank_query": true}'  # ad-hoc

Variants are accumulating: each one inherits the previous best + a new knob.
Add a new entry to VARIANTS, then run the bench.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from rag import engine  # noqa: E402
from experiment import engine as xengine  # noqa: E402

QUESTION_PATH = REPO_ROOT / "question.xlsx"
ANSWER_PATH = REPO_ROOT / "answer.xlsx"
RESULTS_DIR = REPO_ROOT / "experiment"

# Named variants — running list. Add a row, rerun bench.
# Each entry is a CFG OVERLAY on top of xengine.DEFAULT_CONFIG.
VARIANTS: dict[str, dict] = {
    # strict30/pure was best at 9/15. Now try: stronger prompt + ensemble pool
    "strict30/pure":       {"strict_llm_rerank": True, "strict_n": 30,
                            "skip_rerank": True, "retrieve_k": 100},
    "strict80/pure":       {"strict_llm_rerank": True, "strict_n": 80,
                            "skip_rerank": True, "retrieve_k": 100},
    "strict30/ensemble":   {"strict_llm_rerank": True, "ensemble": True,
                            "strict_n": 50, "skip_rerank": True, "retrieve_k": 100},
    "strict50/ensemble":   {"strict_llm_rerank": True, "ensemble": True,
                            "strict_n": 80, "skip_rerank": True, "retrieve_k": 100},
}


# ---------- Scoring (shared with prior experiments) -----------------------

_MODEL_TOKEN_RE = re.compile(r"[A-Z][A-Z0-9-]+")


def parse_gold_models(expected: str) -> list[str]:
    if not expected or expected.strip() == "無匹配產品":
        return []
    out: list[str] = []
    for tok in re.split(r"[+\s,，、/]+", expected):
        m = _MODEL_TOKEN_RE.match(tok.strip())
        if not m:
            continue
        model = m.group(0).rstrip("-")
        if len(model) >= 4 and sum(1 for c in model if c.isupper()) >= 2:
            out.append(model)
    return out


def hits_in(docs: list[dict], gold: list[str]) -> list[str]:
    if not gold:
        return []
    parts: list[str] = []
    for d in docs:
        parts.append(d.get("text", ""))
        meta = d.get("metadata", {})
        parts.append(str(meta.get("models", "")))
        parts.append(str(meta.get("series_name", "")))
        parts.append(d.get("llm_name", ""))
    blob = " ".join(parts).upper()
    return [m for m in gold if m.upper() in blob]


# ---------- Bench runner ---------------------------------------------------


def run_variant(name: str, cfg_overlay: dict,
                questions: list[str], expected: list[str]) -> dict:
    print(f"\n{'='*70}\nVariant: {name}\nCFG overlay: {cfg_overlay}\n{'='*70}")
    rows = []
    for qi, (q, ex) in enumerate(zip(questions, expected)):
        q_clean = q.strip().replace("\n", " ")
        gold = parse_gold_models(ex)
        no_match = ex.strip() == "無匹配產品"
        t0 = time.time()
        try:
            docs = xengine.search(q_clean, top_k=5, cfg=cfg_overlay)
        except Exception as e:
            print(f"Q{qi+1} FAILED: {type(e).__name__}: {e}")
            docs = []
        dt = time.time() - t0
        h5 = hits_in(docs[:5], gold)
        h1 = hits_in(docs[:1], gold)
        models_per_pos = []
        for d in docs[:5]:
            mlist = (d.get("metadata", {}) or {}).get("models", "") or d.get("llm_name", "")
            pg = (d.get("metadata", {}) or {}).get("page", "?")
            models_per_pos.append(f"p{pg}:{mlist[:25]}")
        print(f"Q{qi+1:>2d} t={dt:4.1f}s gold={gold} hit@5={'✓' if h5 else '✗'} "
              f"hit@1={'✓' if h1 else '✗'} | {models_per_pos[0] if models_per_pos else '-'}")
        rows.append({
            "qid": qi + 1, "query": q_clean, "expected": ex.strip(),
            "gold_models": ",".join(gold), "is_no_match": no_match,
            "hit5": bool(h5), "hit1": bool(h1),
            "hits": ",".join(h5), "top5": " | ".join(models_per_pos),
            "latency_s": round(dt, 2),
        })
    scoreable = [r for r in rows if not r["is_no_match"]]
    h5_total = sum(1 for r in scoreable if r["hit5"])
    h1_total = sum(1 for r in scoreable if r["hit1"])
    n = len(scoreable)
    pct = h5_total / n * 100 if n else 0.0
    print(f"\n→ {name}: hit@5 {h5_total}/{n} = {pct:.1f}%  |  hit@1 {h1_total}/{n}  "
          f"|  goal 70% = 11/{n}")
    return {"name": name, "cfg": cfg_overlay, "rows": rows,
            "hit5": h5_total, "hit1": h1_total, "n": n, "pct": pct}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None,
                    help="Named variant (see VARIANTS). Default: all.")
    ap.add_argument("--cfg", default=None,
                    help="Ad-hoc JSON cfg overlay. Overrides --variant.")
    ap.add_argument("--save", action="store_true",
                    help="Save per-question xlsx for the run.")
    args = ap.parse_args()

    print("[1/2] Loading engine state (once for all variants)…")
    t0 = time.time()
    engine.initialize()
    print(f"      ready in {time.time()-t0:.1f}s")

    print("\n[2/2] Loading question.xlsx + answer.xlsx…")
    q_df = pd.read_excel(QUESTION_PATH)
    a_df = pd.read_excel(ANSWER_PATH)
    if len(q_df) != len(a_df):
        raise SystemExit("Length mismatch")
    questions = q_df["詢問問題"].astype(str).tolist()
    expected = a_df["期望回答"].astype(str).tolist()
    print(f"      {len(questions)} questions  "
          f"({sum(1 for e in expected if e.strip()=='無匹配產品')} '無匹配')")

    if args.cfg:
        cfg = json.loads(args.cfg)
        runs = [run_variant("adhoc", cfg, questions, expected)]
    elif args.variant:
        if args.variant not in VARIANTS:
            raise SystemExit(f"Unknown variant. Pick one of: {list(VARIANTS)}")
        runs = [run_variant(args.variant, VARIANTS[args.variant], questions, expected)]
    else:
        runs = [run_variant(name, cfg, questions, expected)
                for name, cfg in VARIANTS.items()]

    # ---- Cross-variant summary ----
    print("\n" + "=" * 70)
    print("Cross-variant summary")
    print("=" * 70)
    print(f"{'Variant':<22s} {'hit@5':>10s} {'hit@1':>10s} {'pct':>8s} {'cfg':<40s}")
    for r in runs:
        print(f"{r['name']:<22s} {r['hit5']:>3d}/{r['n']:<6d} {r['hit1']:>3d}/{r['n']:<6d} "
              f"{r['pct']:>6.1f}%  {json.dumps(r['cfg'], ensure_ascii=False)[:40]}")
    best = max(runs, key=lambda r: (r["pct"], r["hit1"]))
    print(f"\nBest: {best['name']} → {best['pct']:.1f}% "
          f"({'GOAL HIT' if best['pct'] >= 70 else 'goal not met'})")

    if args.save:
        for r in runs:
            out = RESULTS_DIR / f"bench_{r['name'].replace('+','_')}.xlsx"
            pd.DataFrame(r["rows"]).to_excel(out, index=False)
            print(f"  saved → {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
