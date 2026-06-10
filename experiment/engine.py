"""Experimental RAG engine — iterate here, then graduate to rag/engine.py.

Reuses rag.engine for state/init/embeddings/reranker. Exposes one search()
with a CONFIG dict that toggles every experimental knob. Default CONFIG
matches today's rag/engine.py production behaviour, so an unchanged CONFIG
reproduces baseline 5/15 on question.xlsx.

CONFIG keys (additive — set to None / False to disable):
    enrich_rerank_query  : bool        # ✅ C variant, +3 in last experiment
    metadata_filter      : "soft"|"hard"|None
    boost                : float       # soft-filter boost on valid chunks
    penalty              : float       # soft-filter penalty on invalid chunks
    multi_query          : int|None    # k phrasings via gpt-4o-mini; fuse via RRF
    rrf_k                : int         # RRF constant
    model_prefix_boost   : bool        # detect SKU-like tokens in query, BM25 boost
    use_llm_select       : bool        # GPT-4o final pick (default True, matches prod)

Run a one-off variant from CLI:
    python3 -c "from experiment.engine import search; print(search('15W 崁燈 6500K'))"
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

from rag import engine  # noqa: E402
import jieba  # noqa: E402

COLBERT_CACHE = REPO_ROOT / "bge_m3_embeddings" / "chunk_tokens.npz"


# ---------- Config ---------------------------------------------------------

DEFAULT_CONFIG: dict = {
    # ---- Stage 1: hybrid retrieve ----
    "metadata_filter":     "soft",   # "soft" | "hard" | None
    "boost":               1.3,
    "penalty":             0.7,
    "rrf_hybrid":          False,    # fuse BM25 + dense via RRF instead of weighted min-max
    "model_prefix_boost":  False,    # extract SKU-like tokens, add to BM25 query
    "sku_score_boost":     0.0,      # multiplicative boost on chunks whose metadata.models contain an SKU token (0 = off, e.g. 1.5 = +50% on hits)

    # ---- Stage 1b: multi-query fusion ----
    "multi_query":         None,     # None | int (e.g. 3) — gpt-4o-mini paraphrases
    "rrf_k":               60,

    # ---- Stage 1c: query2doc (concat, not replace — distinct from HyDE) ----
    "query2doc":           False,    # gpt-4o-mini generates 80-150 char catalog entry, CONCAT to vec+bm25 query

    # ---- Stage 1d: BGE-M3 ColBERT-style MaxSim 3rd signal ----
    "colbert_signal":      False,    # add MaxSim over BGE-M3 token embeddings
    "colbert_weight":      0.4,      # weight in 3-way fusion (dense+bm25+colbert)

    # ---- Stage 2: BGE rerank ----
    "enrich_rerank_query": False,    # use expand_specs(q) + metadata summary
    "retrieve_k":          50,       # hybrid candidates fetched (engine default 50)
    "rerank_candidates":   20,       # how many to feed BGE reranker (engine default 20)
    "skip_rerank":         False,    # bypass BGE rerank entirely, use hybrid top-N
    "rerank_blend":        0.0,      # 0=pure rerank, 1=pure hybrid; mix the two scores

    # ---- Stage 3: LLM pick ----
    "use_llm_select":      True,
    "strict_llm_rerank":   False,    # use stricter scored LLM prompt over hybrid top-N
    "strict_n":            30,
    "ensemble":            False,    # union top-5 from {hybrid, rrf, rrf+cb}, strict-pick 5
}


# ---------- Helpers --------------------------------------------------------

# SKU-like model number patterns we see in queries / catalog:
#   OD-3204-60, LED-4140R5, E-FLCS50D, D-PD25N-EGR1, DSTMS, L4140R5
_MODEL_TOKEN_RE = re.compile(r"\b[A-Z]{1,4}[A-Z0-9-]{3,}\b")
_HAS_DIGIT_RE = re.compile(r"\d")


def _extract_model_tokens(query: str) -> list[str]:
    """Best-effort SKU-token extraction from a Chinese-mixed query."""
    upper = query.upper()
    out: list[str] = []
    for m in _MODEL_TOKEN_RE.finditer(upper):
        tok = m.group(0).rstrip("-")
        if len(tok) >= 4 and _HAS_DIGIT_RE.search(tok):
            out.append(tok)
    return out


def _render_metadata_summary(specs: dict) -> str:
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


# ---------- Stage 1: parametric hybrid retrieve ----------------------------


def _hybrid_retrieve(query: str, bm25_query: str, top_k: int,
                     valid_indices: set, cfg: dict,
                     sku_tokens: list[str] | None = None) -> list[dict]:
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

    colbert_norm = None
    if cfg["colbert_signal"]:
        colbert_norm = _colbert_scores(query)

    if cfg["rrf_hybrid"]:
        # Rank-based fusion: each list ranks all chunks, RRF combines
        bm25_rank = np.argsort(-bm25_norm)
        vec_rank = np.argsort(-cos_scores)
        k = cfg["rrf_k"]
        rrf = np.zeros(len(chunks), dtype=np.float64)
        for r, i in enumerate(bm25_rank):
            rrf[i] += 1.0 / (k + r + 1)
        for r, i in enumerate(vec_rank):
            rrf[i] += 1.0 / (k + r + 1)
        if colbert_norm is not None:
            for r, i in enumerate(np.argsort(-colbert_norm)):
                rrf[i] += 1.0 / (k + r + 1)
        hybrid = rrf
    else:
        hybrid = engine.BM25_WEIGHT * bm25_norm + engine.VECTOR_WEIGHT * vec_norm
        if colbert_norm is not None:
            # Author-recommended weights [0.4, 0.2, 0.4] for dense/sparse/colbert.
            # We approximate with cfg colbert_weight; renormalise existing two.
            cw = cfg["colbert_weight"]
            scale = 1.0 - cw
            hybrid = scale * hybrid + cw * colbert_norm

    # SKU exact-match score boost on chunks whose metadata.models contain the query SKU
    sku_boost = cfg["sku_score_boost"]
    if sku_boost and sku_tokens:
        toks_upper = [t.upper() for t in sku_tokens]
        for i, m in enumerate(metas):
            blob = (str(m.get("models", "")) + " " + str(m.get("series_name", ""))).upper()
            if any(t in blob for t in toks_upper):
                hybrid[i] *= sku_boost

    mode = cfg["metadata_filter"]
    if mode and valid_indices is not None and len(valid_indices) < len(chunks):
        if mode == "hard" and len(valid_indices) > 0:
            mask = np.zeros(len(chunks), dtype=bool)
            for i in valid_indices:
                mask[i] = True
            hybrid = np.where(mask, hybrid, -1.0)
        elif mode == "soft":
            boost, penalty = cfg["boost"], cfg["penalty"]
            for i in range(len(chunks)):
                hybrid[i] *= boost if i in valid_indices else penalty

    top_idx = np.argsort(hybrid)[::-1][:top_k]
    return [{"text": chunks[i], "metadata": metas[i], "score": float(hybrid[i]),
             "bm25_score": float(bm25_norm[i]), "vector_score": float(vec_norm[i]),
             "_idx": int(i)}
            for i in top_idx]


# ---------- Stage 1b: multi-query + RRF ------------------------------------


def _generate_query2doc(query: str) -> str:
    """gpt-4o-mini → ONE 80-150 char zh-TW catalog entry that matches the user's needs.
    Cached in-memory."""
    cache = _generate_query2doc._cache  # type: ignore[attr-defined]
    if query in cache:
        return cache[query]
    client = engine._get_openai()
    prompt = (
        "你是舞光 (Dancelight) LED 型錄編輯。針對下列客戶需求,寫出一段最可能匹配的型錄條目"
        "(繁體中文,80-150 字),包含型號、瓦數、色溫、光通量、IP 等級、材質與適用場域。"
        "不要解釋,直接輸出條目。\n\n需求: " + query
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    except Exception as e:
        print(f"[query2doc] fail: {type(e).__name__}: {e}")
        text = ""
    cache[query] = text
    return text
_generate_query2doc._cache = {}  # type: ignore[attr-defined]


def _generate_query_paraphrases(query: str, n: int) -> list[str]:
    """gpt-4o-mini → n diverse paraphrases of the user query in zh-TW.
    Cached in-memory for the process life."""
    if n <= 0:
        return []
    cache = _generate_query_paraphrases._cache  # type: ignore[attr-defined]
    if query in cache:
        return cache[query][:n]
    client = engine._get_openai()
    prompt = (
        f"請把下列舞光 LED 產品需求改寫成 {n} 句不同角度的繁體中文檢索 query,"
        f"每句獨立、保留所有規格(W / K / lm / IP / 類別 / 型號片段),"
        f"輸出 JSON: {{\"queries\": [\"...\", \"...\", \"...\"]}}\n\n需求: {query}"
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        qs = [q.strip() for q in data.get("queries", []) if q and q.strip()]
    except Exception as e:
        print(f"[multi_query] fail: {type(e).__name__}: {e}")
        qs = []
    cache[query] = qs
    return qs[:n]
_generate_query_paraphrases._cache = {}  # type: ignore[attr-defined]


def _rrf_fuse(rankings: list[list[dict]], k: int) -> list[dict]:
    """Reciprocal Rank Fusion across multiple ranked lists, keyed by chunk _idx."""
    scores: dict[int, float] = {}
    repr_doc: dict[int, dict] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            idx = doc["_idx"]
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
            if idx not in repr_doc:
                repr_doc[idx] = doc
    fused = sorted(repr_doc.values(),
                   key=lambda d: scores[d["_idx"]], reverse=True)
    for d in fused:
        d["score"] = scores[d["_idx"]]
    return fused


# ---------- Stage 1d: BGE-M3 ColBERT-style MaxSim --------------------------

_colbert_state: dict = {}


def _l2norm_rows(arr: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(arr, axis=1, keepdims=True)
    n = np.where(n > 1e-9, n, 1.0)
    return arr / n


def _build_or_load_colbert() -> None:
    """Build per-chunk token embeddings for MaxSim. Cache to disk."""
    if _colbert_state:
        return
    if COLBERT_CACHE.exists():
        data = np.load(COLBERT_CACHE)
        _colbert_state["all_tokens"] = data["all_tokens"]   # (N_total, 1024) float16
        _colbert_state["offsets"] = data["offsets"]         # (n_chunks+1,) int32
        print(f"[colbert] loaded cache: {_colbert_state['all_tokens'].shape}")
        return
    print("[colbert] cache miss — building per-chunk token embeddings…")
    model = engine._get_query_embedder()
    chunks = engine._state["chunks"]
    all_tok_list: list[np.ndarray] = []
    offsets = [0]
    BATCH = 8
    import time as _t
    t0 = _t.time()
    for i in range(0, len(chunks), BATCH):
        batch = [c[:2000] for c in chunks[i:i+BATCH]]
        tok_lists = model.encode(batch, output_value="token_embeddings",
                                 convert_to_numpy=False, show_progress_bar=False)
        for tok in tok_lists:
            arr = tok.detach().cpu().numpy().astype(np.float32)
            arr = _l2norm_rows(arr)
            all_tok_list.append(arr.astype(np.float16))
            offsets.append(offsets[-1] + arr.shape[0])
        if (i // BATCH) % 5 == 0:
            print(f"  [{i+len(batch)}/{len(chunks)}] {_t.time()-t0:.1f}s")
    all_tokens = np.concatenate(all_tok_list, axis=0)
    offsets_arr = np.asarray(offsets, dtype=np.int32)
    COLBERT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(COLBERT_CACHE, all_tokens=all_tokens, offsets=offsets_arr)
    _colbert_state["all_tokens"] = all_tokens
    _colbert_state["offsets"] = offsets_arr
    print(f"[colbert] built {all_tokens.shape}  total {_t.time()-t0:.1f}s")


def _colbert_query_tokens(query: str) -> np.ndarray:
    model = engine._get_query_embedder()
    toks = model.encode([query[:1000]], output_value="token_embeddings",
                        convert_to_numpy=False, show_progress_bar=False)
    arr = toks[0].detach().cpu().numpy().astype(np.float32)
    return _l2norm_rows(arr)


def _colbert_scores(query: str) -> np.ndarray:
    """Return MaxSim score per chunk (length = n_chunks). Normalised to [0,1]."""
    _build_or_load_colbert()
    q_tok = _colbert_query_tokens(query)            # (q_len, 1024)
    all_tokens = _colbert_state["all_tokens"].astype(np.float32)  # (N_total, 1024)
    offsets = _colbert_state["offsets"]             # (n_chunks+1,)
    # Compute full (q_len, N_total) similarity matrix
    sim = q_tok @ all_tokens.T                      # (q_len, N_total)
    n_chunks = len(offsets) - 1
    scores = np.zeros(n_chunks, dtype=np.float32)
    for i in range(n_chunks):
        s, e = offsets[i], offsets[i+1]
        if s == e:
            continue
        chunk_sim = sim[:, s:e]                     # (q_len, chunk_len)
        # MaxSim: for each query token, max over chunk tokens, then sum
        scores[i] = chunk_sim.max(axis=1).sum()
    # Normalise to [0,1] for fusion
    smin, smax = scores.min(), scores.max()
    return (scores - smin) / (smax - smin + 1e-8)


# ---------- Stage 3 alt: strict-scored LLM rerank --------------------------

STRICT_RERANK_PROMPT = """你是舞光 LED 型錄檢索評選員。客戶用自然語言描述需求,你拿到 {n} 個候選型錄段落,
要選出最匹配的 5 個。請對每個候選逐項評分,再選總分前 5。

【客戶需求】
{query}

【需求解析(輔助,可能不完整)】
{specs_summary}

【評分標準】(每項 0-10 分)
1. **類別匹配**:候選段落的「類別」欄位是否與客戶需求對應的類別一致(同義詞如階梯燈≈步道燈、崁入筒燈≈崁燈算一致;不同產品線給 0 分)。
2. **瓦數匹配**:候選的瓦數是否落在客戶要求範圍內或接近(±30% 算近似)。客戶未指定瓦數時給滿分。
3. **色溫 / 光通量 / IP 等級**:逐項比對,符合得分。客戶未指定者給滿分。
4. **完整型號族**:候選段落含有客戶可能想要的型號族(同一前綴 / 同類功能),加分。
5. **加分項**:防眩、節能標章、感應、調光等若客戶提到才算分。

【硬規則】
- 若客戶明確要求「IP66 或以上」,候選 IP 低於 66 直接 0 分(類別欄位無 IP 訊息者除外)。
- 若候選明顯是品牌頁、目錄索引、實驗室照片,不選。
- 候選有「步道燈、車道燈、戶外、IP」字眼時優先考慮戶外/景觀需求。
- 即使所有候選都不完美,仍要選出 5 個(挑相對最接近的),不要返回少於 5 個。
- **型號前綴差異**:當有多個候選類別相同、瓦數相同、且型號前綴不同(如 LED-、D-、OD- 並存),
  「D-」/「OD-」前綴往往代表較新或特殊系列,「LED-」是舊系列。**請把不同前綴的同類產品都放入 top-5**,
  不要只挑 LED- 變體而忽略 D-/OD- 型號族。
- 客戶提到「樓梯」/「階梯」時,優先 IP65+ 的崁燈/步道燈,避免一般室內崁燈。
- 客戶提到「玻璃燈罩」時,優先含「玻璃」、「燈罩」、「索爾」、「黑鑽」等關鍵詞的型號。

【輸出格式】JSON,picks 陣列 5 筆,從最佳到次佳:
{{
  "picks": [
    {{"doc_id": <1~{n}>, "name": "<產品系列名>", "score_breakdown": "類別X 瓦數X IP X ...", "reason": "<25 字內>"}},
    ...
  ]
}}

【候選段落】
{context}"""


def _strict_llm_rerank(query: str, specs: dict, candidates: list[dict],
                        top_k: int, n: int) -> list[dict]:
    short = candidates[:n]
    if not short:
        return []
    specs_lines = []
    for k, v in specs.items():
        specs_lines.append(f"{k}={v}")
    specs_summary = "; ".join(specs_lines) if specs_lines else "(無 spec 抽取)"
    context = engine._build_llm_context(short, per_doc=900)
    prompt = STRICT_RERANK_PROMPT.format(n=len(short), query=query,
                                          specs_summary=specs_summary,
                                          context=context)
    client = engine._get_openai()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = json.loads(resp.choices[0].message.content or "{}")
        picks = data.get("picks", [])
    except Exception as e:
        print(f"[strict_llm_rerank] fail: {type(e).__name__}: {e}")
        return short[:top_k]

    seen = set()
    out = []
    for p in picks:
        doc_id = p.get("doc_id")
        if not isinstance(doc_id, int) or doc_id < 1 or doc_id > len(short) or doc_id in seen:
            continue
        seen.add(doc_id)
        c2 = dict(short[doc_id - 1])
        c2["rank_label"] = "★ 推薦" if not out else f"備選 {len(out)}"
        c2["llm_reason"] = (p.get("reason") or "").strip()
        c2["llm_name"] = (p.get("name") or "").strip()
        c2["llm_breakdown"] = (p.get("score_breakdown") or "").strip()
        out.append(c2)
        if len(out) >= top_k:
            break
    # pad with rerank order if LLM returned fewer than top_k
    for i, c in enumerate(short):
        if len(out) >= top_k:
            break
        if (i + 1) in seen:
            continue
        c2 = dict(c)
        c2["rank_label"] = "★ 推薦" if not out else f"備選 {len(out)}"
        c2["llm_reason"] = "(LLM 未挑選,補位)"
        out.append(c2)
    return out


# ---------- Stage 2: BGE rerank with configurable query --------------------


def _bge_rerank(query_for_rerank: str, candidates: list[dict],
                top_k: int, *, n_rerank: int, blend: float = 0.0) -> list[dict]:
    """BGE rerank with configurable shortlist depth and optional hybrid-blend.

    blend=0.0 → pure rerank score
    blend=1.0 → pure hybrid score (no rerank used in ordering)
    blend=0.5 → average of normalised rerank + hybrid score
    """
    shortlist = candidates[:n_rerank]
    if not shortlist:
        return []
    reranker = engine._get_reranker()
    pairs = [(query_for_rerank, c["text"][:2000]) for c in shortlist]
    try:
        scores = reranker.predict(pairs).tolist()
    except Exception as e:
        print(f"[BGE Rerank] failed: {e}")
        for c in candidates:
            c["rerank_score"] = c.get("score", 0.0)
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates[:top_k]
    for c, s in zip(shortlist, scores):
        c["rerank_score"] = float(s)
    for c in candidates[n_rerank:]:
        c["rerank_score"] = float(c.get("score", 0.0))

    if blend > 0.0:
        r = np.asarray([c["rerank_score"] for c in shortlist], dtype=np.float64)
        h = np.asarray([c["score"] for c in shortlist], dtype=np.float64)
        rmin, rmax = r.min(), r.max()
        hmin, hmax = h.min(), h.max()
        rn = (r - rmin) / (rmax - rmin + 1e-8)
        hn = (h - hmin) / (hmax - hmin + 1e-8)
        blended = (1 - blend) * rn + blend * hn
        for c, b in zip(shortlist, blended):
            c["rerank_score"] = float(b)
        for c in candidates[n_rerank:]:
            c["rerank_score"] = 0.0  # demote out-of-pool

    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


# ---------- Top-level search ----------------------------------------------


def search(query: str, *, top_k: int = 5, cfg: dict | None = None) -> list[dict]:
    """Run the configured experimental pipeline. cfg overrides DEFAULT_CONFIG keys."""
    if not engine._state:
        engine.initialize()
    engine._load_annotations()

    config = dict(DEFAULT_CONFIG)
    if cfg:
        config.update(cfg)

    specs = engine._decompose_query(query)
    valid = engine._metadata_filter(specs, engine._state["chunk_metas"])
    expanded = engine._expand_specs(query)
    bm25_q = engine._add_synonyms(query)

    sku_toks = _extract_model_tokens(query)
    if config["model_prefix_boost"] and sku_toks:
        for tok in sku_toks:
            bm25_q = f"{bm25_q} {tok}"

    # Query2doc: append generated pseudo-document to BOTH bm25 and vec query
    q2d_text = ""
    if config["query2doc"]:
        q2d_text = _generate_query2doc(query)
        if q2d_text:
            expanded = f"{expanded}\n[Q2D]\n{q2d_text}"
            bm25_q = f"{bm25_q} {q2d_text}"

    retrieve_k = config["retrieve_k"]
    # ---- Stage 1: hybrid (single or multi-query) ----
    if config["multi_query"]:
        paras = _generate_query_paraphrases(query, config["multi_query"])
        queries = [query] + paras
        rankings = []
        for q in queries:
            q_expanded = engine._expand_specs(q)
            q_bm25 = engine._add_synonyms(q)
            q_skus = _extract_model_tokens(q)
            if config["model_prefix_boost"] and q_skus:
                for tok in q_skus:
                    q_bm25 = f"{q_bm25} {tok}"
            if q2d_text:
                q_expanded = f"{q_expanded}\n[Q2D]\n{q2d_text}"
                q_bm25 = f"{q_bm25} {q2d_text}"
            r = _hybrid_retrieve(q_expanded, q_bm25, retrieve_k,
                                 valid, config, sku_tokens=sku_toks)
            rankings.append(r)
        candidates = _rrf_fuse(rankings, k=config["rrf_k"])[:retrieve_k]
    else:
        candidates = _hybrid_retrieve(expanded, bm25_q, retrieve_k,
                                      valid, config, sku_tokens=sku_toks)

    # ---- Stage 2: BGE rerank (optional) ----
    if config["skip_rerank"]:
        # Feed up to config['rerank_candidates'] to LLM (default 20)
        reranked = candidates[:config["rerank_candidates"]]
        for c in reranked:
            c["rerank_score"] = c.get("score", 0.0)
    else:
        if config["enrich_rerank_query"]:
            meta = _render_metadata_summary(specs)
            rerank_q = f"{expanded}\n{meta}" if meta else expanded
        else:
            rerank_q = query
        reranked = _bge_rerank(rerank_q, candidates,
                               top_k=engine.LLM_SELECT_CANDIDATES,
                               n_rerank=config["rerank_candidates"],
                               blend=config["rerank_blend"])

    # ---- Stage 3: LLM pick (optional) ----
    if config["ensemble"]:
        # Build union of top-N from 3 different hybrid configs.
        configs = [
            {},
            {"rrf_hybrid": True},
            {"rrf_hybrid": True, "colbert_signal": True},
        ]
        seen = {}
        for overlay in configs:
            cfg2 = dict(config); cfg2.update(overlay)
            sku_ts = _extract_model_tokens(query)
            r = _hybrid_retrieve(expanded, bm25_q, retrieve_k,
                                  valid, cfg2, sku_tokens=sku_ts)
            for c in r[:30]:
                if c["_idx"] not in seen:
                    seen[c["_idx"]] = c
        pool = list(seen.values())
        return _strict_llm_rerank(query, specs, pool,
                                  top_k=top_k, n=min(len(pool), config["strict_n"]))
    if config["strict_llm_rerank"]:
        pool = candidates if config["skip_rerank"] else reranked
        return _strict_llm_rerank(query, specs, pool,
                                  top_k=top_k, n=config["strict_n"])
    if config["use_llm_select"]:
        return engine._llm_select(query, reranked, top_k=top_k)
    return reranked[:top_k]
