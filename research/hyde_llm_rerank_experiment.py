"""HyDE / LLM-rerank ablation on Training.xlsx.

Five variants compared:
  A.        baseline                : hybrid → BGE rerank → top-5
  B-mini.   +HyDE (gpt-4o-mini)     : HyDE → hybrid → BGE rerank → top-5
  B-local.  +HyDE (qwen3.6 Ollama)  : HyDE → hybrid → BGE rerank → top-5
  C-mini.   +HyDE-mini +LLM-rerank  : HyDE-mini → hybrid → BGE rerank → gpt-4o picks 5
  C-local.  +HyDE-local +LLM-rerank : HyDE-local → hybrid → BGE rerank → gpt-4o picks 5

See specs/hyde-llm-rerank-experiment.md for context.
Run: cd <repo> && python3 -m research.hyde_llm_rerank_experiment
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

# Make sure we resolve repo root regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from rag import engine  # noqa: E402

TRAINING_PATH = REPO_ROOT / "Training.xlsx"
HYDE_CACHE = REPO_ROOT / "research" / ".hyde_cache.json"
RESULTS_XLSX = REPO_ROOT / "research" / "hyde_llm_rerank_results.xlsx"

OPENAI_HYDE_MODEL = "gpt-4o-mini"
OLLAMA_HYDE_MODEL = "qwen3.6:latest"
TOP_K = 5

HYDE_PROMPT = """你是舞光 (Dancelight) LED 照明型錄編輯。根據下列客戶需求,寫出 ONE 條最匹配的型錄條目 (繁體中文,80-150 字),
條目需包含: 產品系列名、型號 (像 E-FLCS50D / D-21DOP25NR2 這樣的字母+數字+橫線格式)、瓦數、色溫、
光通量 (lm)、IP 等級、本體材質、適用場域。即使需求模糊也要寫出具體的型錄文字,不要解釋,直接給條目。

客戶需求:
{query}

型錄條目:"""


# ---------- HyDE generation -------------------------------------------------

_hyde_cache: dict = {}


def _load_hyde_cache() -> None:
    global _hyde_cache
    if HYDE_CACHE.exists():
        _hyde_cache = json.loads(HYDE_CACHE.read_text())
    else:
        _hyde_cache = {}


def _save_hyde_cache() -> None:
    HYDE_CACHE.write_text(json.dumps(_hyde_cache, ensure_ascii=False, indent=2))


def _hyde_key(query: str, provider: str) -> str:
    return f"{provider}:{hashlib.md5(query.encode('utf-8')).hexdigest()}"


def hyde_openai(query: str) -> str:
    """HyDE via gpt-4o-mini. Cached."""
    key = _hyde_key(query, "openai-mini")
    if key in _hyde_cache:
        return _hyde_cache[key]
    client = engine._get_openai()
    resp = client.chat.completions.create(
        model=OPENAI_HYDE_MODEL,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    _hyde_cache[key] = text
    _save_hyde_cache()
    return text


def hyde_ollama(query: str) -> str:
    """HyDE via local qwen3.6. Cached."""
    key = _hyde_key(query, "ollama-qwen36")
    if key in _hyde_cache:
        return _hyde_cache[key]
    import ollama  # local import; only needed for B-local / C-local
    resp = ollama.chat(
        model=OLLAMA_HYDE_MODEL,
        messages=[{"role": "user", "content": HYDE_PROMPT.format(query=query)}],
        options={"temperature": 0, "num_ctx": 2048, "num_predict": 300},
    )
    text = (resp["message"]["content"] or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    _hyde_cache[key] = text
    _save_hyde_cache()
    return text


# ---------- Retrieval pipeline (parametric) ---------------------------------


def retrieve(query: str, *, hyde_text: str | None, use_llm_rerank: bool, top_k: int = TOP_K) -> list[dict]:
    """Run one retrieval variant.

    - hyde_text: prepended to the vector query (along with expand_specs result).
                 If None, only expand_specs is used.
    - use_llm_rerank: if True, pass BGE top-20 through engine._llm_select (gpt-4o picks 5).
                     If False, return BGE-reranked top-k directly.
    """
    specs = engine._decompose_query(query)
    valid = engine._metadata_filter(specs, engine._state["chunk_metas"])
    expanded = engine._expand_specs(query)
    bm25_q = engine._add_synonyms(query)

    vec_q = expanded
    if hyde_text:
        vec_q = f"{expanded}\n\n[HyDE]\n{hyde_text}"

    cands = engine._hybrid_retrieve(vec_q, bm25_q, engine.RETRIEVE_K, valid)
    reranked = engine._bge_rerank(query, cands, top_k=engine.LLM_SELECT_CANDIDATES)

    if use_llm_rerank:
        return engine._llm_select(query, reranked, top_k=top_k)
    return reranked[:top_k]


# ---------- Scoring ---------------------------------------------------------

_MODEL_TOKEN_RE = re.compile(r"[A-Z][A-Z0-9-]+")


def parse_gold_models(expected: str) -> list[str]:
    """Extract model number tokens from a Training answer.

    Handles: "E-FLCS50D+燈具相關資訊(...)"  →  ["E-FLCS50D"]
             "LED-2441R1+D-T810DR9"           →  ["LED-2441R1", "D-T810DR9"]
             "L4140R5+DSTMS"                  →  ["L4140R5", "DSTMS"]
             "無匹配產品"                      →  []
    """
    if not expected or expected.strip() == "無匹配產品":
        return []
    out: list[str] = []
    for tok in re.split(r"[+\s,，、/]+", expected):
        m = _MODEL_TOKEN_RE.match(tok.strip())
        if not m:
            continue
        model = m.group(0).rstrip("-")
        # Must be >=4 chars AND >=2 uppercase letters (filter out chinese-desc noise).
        if len(model) >= 4 and sum(1 for c in model if c.isupper()) >= 2:
            out.append(model)
    return out


def hits_in(docs: list[dict], gold_models: list[str]) -> list[str]:
    """Return which gold models appear in the candidates' text/metadata."""
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


# ---------- Main driver -----------------------------------------------------


VARIANTS = [
    ("A.baseline",       {"hyde": None,    "llm_rerank": False}),
    ("B.+HyDE-mini",     {"hyde": "mini",  "llm_rerank": False}),
    ("B.+HyDE-local",    {"hyde": "local", "llm_rerank": False}),
    ("C.+HyDE-mini+LLM", {"hyde": "mini",  "llm_rerank": True}),
    ("C.+HyDE-local+LLM",{"hyde": "local", "llm_rerank": True}),
]


def main() -> None:
    print("=" * 70)
    print("HyDE / LLM-rerank ablation — Training.xlsx (10 questions × 5 variants)")
    print("=" * 70)

    _load_hyde_cache()

    print("\n[1/3] Loading RAG engine (BGE-M3 + BGE-reranker)...")
    t0 = time.time()
    engine.initialize()
    print(f"      engine ready in {time.time()-t0:.1f}s")

    print("\n[2/3] Loading Training.xlsx...")
    df = pd.read_excel(TRAINING_PATH)
    questions = df["詢問問題"].astype(str).tolist()
    expected = df["期望回答"].astype(str).tolist()
    print(f"      {len(questions)} questions")

    print("\n[3/3] Running variants...\n")
    per_question: list[dict] = []

    for qi, (q, ex) in enumerate(zip(questions, expected)):
        q_clean = q.strip().replace("\n", " ")
        gold = parse_gold_models(ex)
        is_no_match = ex.strip() == "無匹配產品"
        print(f"\nQ{qi+1}: {q_clean[:70]}")
        print(f"   expected: {ex.strip()[:60]}  gold_models={gold}")

        # Pre-generate HyDE passages once per question (cached on disk).
        hyde_passages = {"mini": None, "local": None}
        hyde_passages["mini"] = hyde_openai(q_clean)
        try:
            hyde_passages["local"] = hyde_ollama(q_clean)
        except Exception as e:
            print(f"   [HyDE-local FAIL] {type(e).__name__}: {e}")
            hyde_passages["local"] = ""

        row = {
            "qid": qi + 1,
            "query": q_clean,
            "expected": ex.strip(),
            "gold_models": ",".join(gold),
            "is_no_match": is_no_match,
            "hyde_mini": hyde_passages["mini"][:200],
            "hyde_local": hyde_passages["local"][:200],
        }

        for vname, opts in VARIANTS:
            hyde_text = hyde_passages[opts["hyde"]] if opts["hyde"] else None
            t1 = time.time()
            try:
                docs = retrieve(q_clean, hyde_text=hyde_text, use_llm_rerank=opts["llm_rerank"], top_k=TOP_K)
            except Exception as e:
                print(f"   [{vname}] FAILED: {type(e).__name__}: {e}")
                docs = []
            dt = time.time() - t1

            top5_models = []
            for d in docs[:TOP_K]:
                mlist = (d.get("metadata", {}) or {}).get("models", "") or d.get("llm_name", "")
                pg = (d.get("metadata", {}) or {}).get("page", "?")
                top5_models.append(f"p{pg}:{mlist[:30]}")
            hits5 = hits_in(docs[:5], gold)
            hits1 = hits_in(docs[:1], gold)
            llm_declined = False
            if opts["llm_rerank"] and docs:
                joined = " ".join((d.get("llm_reason", "") or "") for d in docs)
                llm_declined = "無匹配" in joined or "no match" in joined.lower()

            print(f"   {vname:20s} t={dt:4.1f}s  hit@5={'✓' if hits5 else '✗'} hit@1={'✓' if hits1 else '✗'}  top1: {top5_models[0] if top5_models else '-'}")

            row[f"{vname}.hit5"] = bool(hits5)
            row[f"{vname}.hit1"] = bool(hits1)
            row[f"{vname}.hits"] = ",".join(hits5)
            row[f"{vname}.top5"] = " | ".join(top5_models)
            row[f"{vname}.llm_declined"] = llm_declined if opts["llm_rerank"] else ""

        per_question.append(row)

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    scoreable = [r for r in per_question if not r["is_no_match"]]
    no_match_rows = [r for r in per_question if r["is_no_match"]]

    print(f"\nScoreable rows (with gold model): {len(scoreable)}")
    print(f"No-match rows:                    {len(no_match_rows)}\n")

    header = f"{'Variant':<20s} {'hit@5':>8s} {'hit@1':>8s} {'no-match-handled':>18s}"
    print(header)
    print("-" * len(header))
    for vname, opts in VARIANTS:
        h5 = sum(1 for r in scoreable if r[f"{vname}.hit5"])
        h1 = sum(1 for r in scoreable if r[f"{vname}.hit1"])
        if opts["llm_rerank"]:
            nm = sum(1 for r in no_match_rows if r[f"{vname}.llm_declined"])
            nm_str = f"{nm}/{len(no_match_rows)}"
        else:
            nm_str = "n/a"
        print(f"{vname:<20s} {h5:>3d}/{len(scoreable):<4d} {h1:>3d}/{len(scoreable):<4d} {nm_str:>18s}")

    # Per-question hit grid
    print("\nPer-question hit@5 grid (rows=Q, cols=variant):")
    print(f"{'Q':<4s}{'gold':<25s}" + "".join(f"{v.split('.')[0]:>6s}" for v, _ in VARIANTS))
    for r in per_question:
        marks = "".join(
            f"{'✓' if r[f'{v}.hit5'] else ('-' if r['is_no_match'] else '✗'):>6s}"
            for v, _ in VARIANTS
        )
        gold = r["gold_models"] or "(無匹配)"
        print(f"{r['qid']:<4d}{gold[:24]:<25s}{marks}")

    # Save xlsx
    out_df = pd.DataFrame(per_question)
    out_df.to_excel(RESULTS_XLSX, index=False)
    print(f"\nFull results → {RESULTS_XLSX.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
