"""BGE-rerank query enrichment + metadata-filter strength tuning.

Four variants compared on question.xlsx (17 questions) with gold in answer.xlsx:

  A. baseline            : raw query → BGE rerank ;  filter 1.3×/0.7×
  B. filter-stronger     : raw query → BGE rerank ;  filter 1.5×/0.5×
  C. enrich-rerank       : (expand_specs(q) + metadata-summary) → BGE rerank ;  filter 1.3×/0.7×
  D. hard-filter+enrich  : (expand_specs(q) + metadata-summary) → BGE rerank ;  hard filter (drop invalid)

See specs/rerank-query-and-metadata-filter-tuning.md for context.
Run:  cd <repo> && python3 -m research.rerank_metadata_tuning_experiment
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from rag import engine  # noqa: E402
import jieba  # noqa: E402

QUESTION_PATH = REPO_ROOT / "question.xlsx"
ANSWER_PATH = REPO_ROOT / "answer.xlsx"
RESULTS_XLSX = REPO_ROOT / "research" / "rerank_metadata_tuning_results.xlsx"
TOP_K = 5


# ---------- Parametric hybrid retrieve --------------------------------------


def hybrid_retrieve(query: str, bm25_query: str, top_k: int, valid_indices: set,
                    *, boost: float, penalty: float, hard: bool) -> list[dict]:
    """Mirror of engine._hybrid_retrieve with parameterised filter strength.

    - boost / penalty : multiplier for chunks IN / OUT of valid_indices (soft).
    - hard            : if True, restrict candidates to valid_indices entirely
                        (boost / penalty ignored, but if valid_indices is empty
                        we fall back to no filter to avoid empty results).
    """
    bm25 = engine._state["bm25"]
    chunks = engine._state["chunks"]
    metas = engine._state["chunk_metas"]
    embs = engine._state["chunk_embeddings"]

    qt = list(jieba.cut(bm25_query))
    bm25_scores = bm25.get_scores(qt)
    bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1
    bm25_norm = bm25_scores / bm25_max

    q_emb = engine._embed_query(query)
    cos_scores = (embs @ q_emb).flatten()
    vec_min, vec_max = cos_scores.min(), cos_scores.max()
    vec_norm = (cos_scores - vec_min) / (vec_max - vec_min + 1e-8)

    hybrid = engine.BM25_WEIGHT * bm25_norm + engine.VECTOR_WEIGHT * vec_norm

    if valid_indices is not None and len(valid_indices) < len(chunks):
        if hard and len(valid_indices) > 0:
            mask = np.zeros(len(chunks), dtype=bool)
            for i in valid_indices:
                mask[i] = True
            hybrid = np.where(mask, hybrid, -1.0)
        else:
            for i in range(len(chunks)):
                hybrid[i] *= boost if i in valid_indices else penalty

    top_idx = np.argsort(hybrid)[::-1][:top_k]
    return [{"text": chunks[i], "metadata": metas[i], "score": float(hybrid[i]),
             "bm25_score": float(bm25_norm[i]), "vector_score": float(vec_norm[i])}
            for i in top_idx]


# ---------- Parametric BGE rerank -------------------------------------------


def bge_rerank_with_query(rerank_query: str, candidates: list[dict],
                          top_k: int) -> list[dict]:
    """Mirror engine._bge_rerank but the cross-encoder sees `rerank_query`
    instead of the raw user query."""
    shortlist = candidates[:engine.RERANK_CANDIDATES]
    if not shortlist:
        return []
    reranker = engine._get_reranker()
    pairs = [(rerank_query, c["text"][:2000]) for c in shortlist]
    try:
        scores = reranker.predict(pairs).tolist()
    except Exception as e:
        print(f"[BGE Rerank] failed, falling back to hybrid score: {e}")
        for c in candidates:
            c["rerank_score"] = c["score"]
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates[:top_k]
    for c, s in zip(shortlist, scores):
        c["rerank_score"] = float(s)
    for c in candidates[engine.RERANK_CANDIDATES:]:
        c["rerank_score"] = float(c["score"])
    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


# ---------- Metadata-summary string used by C/D rerank query ----------------


def metadata_summary(specs: dict) -> str:
    """Render specs dict as short Chinese key-value lines for the cross-encoder."""
    if not specs:
        return ""
    parts = []
    if specs.get("category"):
        parts.append(f"[類別] {specs['category']}")
    if specs.get("max_wattage") is not None:
        parts.append(f"[最大瓦數] {specs['max_wattage']}W")
    if specs.get("color_temp") is not None:
        parts.append(f"[色溫] {specs['color_temp']}K")
    if specs.get("min_lumens") is not None:
        parts.append(f"[最小光通量] {specs['min_lumens']}lm")
    if specs.get("ip_rating") is not None:
        parts.append(f"[IP] IP{specs['ip_rating']}")
    return "\n".join(parts)


def build_rerank_query(query: str, specs: dict, *, enrich: bool) -> str:
    if not enrich:
        return query
    expanded = engine._expand_specs(query)
    meta = metadata_summary(specs)
    if meta:
        return f"{expanded}\n{meta}"
    return expanded


# ---------- Variant runner --------------------------------------------------


def run_variant(query: str, *, enrich_rerank: bool, boost: float, penalty: float,
                hard: bool, top_k: int = TOP_K, retrieve_k: int = engine.RETRIEVE_K
                ) -> tuple[list[dict], list[dict], list[dict]]:
    """Returns (hybrid_top_retrieve_k, rerank_top_RERANK_CANDIDATES, final_top_k)."""
    specs = engine._decompose_query(query)
    valid = engine._metadata_filter(specs, engine._state["chunk_metas"])
    expanded = engine._expand_specs(query)
    bm25_q = engine._add_synonyms(query)

    cands = hybrid_retrieve(expanded, bm25_q, retrieve_k, valid,
                            boost=boost, penalty=penalty, hard=hard)
    rerank_q = build_rerank_query(query, specs, enrich=enrich_rerank)
    reranked_full = bge_rerank_with_query(rerank_q, cands,
                                          top_k=engine.RERANK_CANDIDATES)
    return cands, reranked_full, reranked_full[:top_k]


# ---------- Scoring ---------------------------------------------------------

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


def hits_in(docs: list[dict], gold_models: list[str]) -> list[str]:
    if not gold_models:
        return []
    parts = []
    for d in docs:
        parts.append(d.get("text", ""))
        meta = d.get("metadata", {})
        parts.append(str(meta.get("models", "")))
        parts.append(str(meta.get("series_name", "")))
        parts.append(d.get("llm_name", ""))
    combined = " ".join(parts).upper()
    return [m for m in gold_models if m.upper() in combined]


# ---------- Main ------------------------------------------------------------

VARIANTS = [
    ("A.baseline",       {"enrich": False, "boost": 1.3, "penalty": 0.7, "hard": False}),
    ("B.filter-stronger",{"enrich": False, "boost": 1.5, "penalty": 0.5, "hard": False}),
    ("C.enrich-rerank",  {"enrich": True,  "boost": 1.3, "penalty": 0.7, "hard": False}),
    ("D.hard+enrich",    {"enrich": True,  "boost": 1.3, "penalty": 0.7, "hard": True}),
]


def main() -> None:
    print("=" * 76)
    print("Rerank-query enrichment + metadata-filter tuning")
    print("question.xlsx (17 Q) × 4 variants")
    print("=" * 76)

    print("\n[1/3] Loading RAG engine (BGE-M3 + BGE-reranker)...")
    t0 = time.time()
    engine.initialize()
    print(f"      engine ready in {time.time()-t0:.1f}s")

    print("\n[2/3] Loading question.xlsx + answer.xlsx...")
    q_df = pd.read_excel(QUESTION_PATH)
    a_df = pd.read_excel(ANSWER_PATH)
    if len(q_df) != len(a_df):
        raise SystemExit(f"Length mismatch: questions={len(q_df)}, answers={len(a_df)}")
    questions = q_df["詢問問題"].astype(str).tolist()
    expected = a_df["期望回答"].astype(str).tolist()
    print(f"      {len(questions)} questions")

    print("\n[3/3] Running variants...\n")
    rows: list[dict] = []
    reachable_at: dict[str, dict[int, int]] = {v: {} for v, _ in VARIANTS}
    # reachable_at[variant][depth] = count of questions whose gold is in
    # hybrid_top_K (depth ∈ {20, 50}) or rerank_top_K (depth ∈ {5, 10, 20}).

    for qi, (q, ex) in enumerate(zip(questions, expected)):
        q_clean = q.strip().replace("\n", " ")
        gold = parse_gold_models(ex)
        is_no_match = ex.strip() == "無匹配產品"
        print(f"\nQ{qi+1}: {q_clean[:70]}")
        print(f"   expected: {ex.strip()[:60]}  gold={gold}")

        row = {
            "qid": qi + 1,
            "query": q_clean,
            "expected": ex.strip(),
            "gold_models": ",".join(gold),
            "is_no_match": is_no_match,
        }

        for vname, opts in VARIANTS:
            t1 = time.time()
            try:
                cands50, reranked20, top5 = run_variant(
                    q_clean,
                    enrich_rerank=opts["enrich"], boost=opts["boost"],
                    penalty=opts["penalty"], hard=opts["hard"],
                )
            except Exception as e:
                print(f"   [{vname}] FAILED: {type(e).__name__}: {e}")
                cands50, reranked20, top5 = [], [], []
            dt = time.time() - t1

            h5 = hits_in(top5, gold)
            h1 = hits_in(top5[:1], gold)
            h10 = hits_in(reranked20[:10], gold)
            h20_rerank = hits_in(reranked20, gold)  # all 20 rerank candidates
            h50_hybrid = hits_in(cands50, gold)     # hybrid top-50

            top5_models = []
            for d in top5:
                mlist = (d.get("metadata", {}) or {}).get("models", "") or d.get("llm_name", "")
                pg = (d.get("metadata", {}) or {}).get("page", "?")
                top5_models.append(f"p{pg}:{mlist[:25]}")

            rerank_score = top5[0].get("rerank_score", 0.0) if top5 else 0.0
            print(f"   {vname:20s} t={dt:4.1f}s "
                  f"hit@5={'✓' if h5 else '✗'} @10={'✓' if h10 else '✗'} "
                  f"@20rr={'✓' if h20_rerank else '✗'} @50hy={'✓' if h50_hybrid else '✗'} "
                  f"r0={rerank_score:+.2f}  top1: {top5_models[0] if top5_models else '-'}")

            row[f"{vname}.hit5"] = bool(h5)
            row[f"{vname}.hit1"] = bool(h1)
            row[f"{vname}.hit10"] = bool(h10)
            row[f"{vname}.hit20rerank"] = bool(h20_rerank)
            row[f"{vname}.hit50hybrid"] = bool(h50_hybrid)
            row[f"{vname}.hits"] = ",".join(h5)
            row[f"{vname}.top5"] = " | ".join(top5_models)

        rows.append(row)

    # ---------- Summary ----------
    print("\n" + "=" * 76)
    print("Summary")
    print("=" * 76)

    scoreable = [r for r in rows if not r["is_no_match"]]
    no_match_rows = [r for r in rows if r["is_no_match"]]
    print(f"\nQuestions with gold model: {len(scoreable)}")
    print(f"'無匹配產品' questions:     {len(no_match_rows)}")

    header = (f"\n{'Variant':<22s} {'hit@5':>9s} {'hit@1':>9s} "
              f"{'@10 rerank':>11s} {'@20 rerank':>11s} {'@50 hybrid':>11s}")
    print(header)
    print("-" * len(header))
    for vname, _ in VARIANTS:
        h5 = sum(1 for r in scoreable if r[f"{vname}.hit5"])
        h1 = sum(1 for r in scoreable if r[f"{vname}.hit1"])
        h10 = sum(1 for r in scoreable if r[f"{vname}.hit10"])
        h20 = sum(1 for r in scoreable if r[f"{vname}.hit20rerank"])
        h50 = sum(1 for r in scoreable if r[f"{vname}.hit50hybrid"])
        n = len(scoreable)
        print(f"{vname:<22s} {h5:>3d}/{n:<5d} {h1:>3d}/{n:<5d} "
              f"{h10:>3d}/{n:<6d} {h20:>3d}/{n:<6d} {h50:>3d}/{n:<6d}")

    # Per-question grid
    print(f"\n{'Per-question hit@5':<78s}")
    print(f"{'Q':<4s}{'gold':<30s}" + "".join(f"{v.split('.')[0]:>5s}" for v, _ in VARIANTS))
    for r in rows:
        marks = "".join(
            f"{'✓' if r[f'{v}.hit5'] else ('-' if r['is_no_match'] else '✗'):>5s}"
            for v, _ in VARIANTS
        )
        gold = r["gold_models"] or "(無匹配)"
        print(f"{r['qid']:<4d}{gold[:29]:<30s}{marks}")

    # Diff vs baseline
    print("\nPer-variant delta vs A.baseline (hit@5 newly gained / lost):")
    base_hits = {r["qid"] for r in scoreable if r["A.baseline.hit5"]}
    for vname, _ in VARIANTS[1:]:
        v_hits = {r["qid"] for r in scoreable if r[f"{vname}.hit5"]}
        gained = sorted(v_hits - base_hits)
        lost = sorted(base_hits - v_hits)
        print(f"   {vname:<22s} +{len(gained)} (Q{gained})   -{len(lost)} (Q{lost})")

    # Save
    out_df = pd.DataFrame(rows)
    out_df.to_excel(RESULTS_XLSX, index=False)
    print(f"\nFull results → {RESULTS_XLSX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
