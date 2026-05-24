"""RAG engine for 舞光 LED catalog — v7 stack.

Pipeline (matches dancelight_rag2.0_2026-04-29/dancelight_rag_test.py @ Apr 30 v7):
  - Chunking: per-page structured (regex-extracted models / W / K / lm / IP /
    features) with image descriptions appended.
  - Embeddings: BAAI/bge-m3 (1024-dim), loaded from cached
    ./bge_m3_embeddings/chunk_embeddings.npy.
  - Retrieval: BM25 + dense hybrid with synonym expansion.
  - Reranker: BAAI/bge-reranker-v2-m3 CrossEncoder (no Ollama LLM in the path).
"""
import json
import os
import re

import fitz
import jieba
import numpy as np
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv(".env")

PDF_PATH = "./2025舞光LED21st(單頁水印可搜尋).pdf"
ODL_JSON = "./output_opendataloader/2025舞光LED21st(單頁水印可搜尋).json"
ODL_DIR = "./output_opendataloader"
IMG_CACHE_FILE = "./img_descriptions_cache.json"
EMBED_CACHE = "./bge_m3_embeddings/chunk_embeddings.npy"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
# Kept for web UI label compatibility (templates/index.html reads rag_engine.LOCAL_LLM)
LOCAL_LLM = RERANK_MODEL

MIN_IMG_SIZE = 30_000
RETRIEVE_K = 50
RERANK_CANDIDATES = 20
BM25_WEIGHT = 0.5
VECTOR_WEIGHT = 0.5

_state: dict = {}
_query_embed_model = None
_reranker = None


LAMP_TERMS = [
    "投射燈", "泛光燈", "崁燈", "筒燈", "軌道燈", "支架燈", "壁燈",
    "吸頂燈", "吊燈", "平板燈", "工事燈", "中東燈", "防潮燈",
    "步道燈", "車道燈", "高燈", "景觀燈", "路燈", "洗牆燈",
    "日光燈", "燈管", "燈泡", "燈座", "燈具", "燈殼", "輕鋼架燈",
    "色溫", "光通量", "演色性", "防水", "節能標章", "全電壓",
    "感應型", "調光", "防眩", "崁入孔", "發光角度", "光束角",
]

SYNONYMS = {
    "投射燈": ["泛光燈", "探照燈", "聚光燈"],
    "泛光燈": ["投射燈", "探照燈"],
    "車道燈": ["步道燈", "走道燈", "戶外崁燈", "引導燈"],
    "步道燈": ["車道燈", "走道燈", "引導燈"],
    "嵌入式日光燈": ["輕鋼架燈", "T-BAR燈", "格柵燈"],
    "日光燈": ["輕鋼架燈", "T-BAR燈", "格柵燈", "支架燈"],
    "崁入筒燈": ["崁燈", "筒燈", "嵌燈"],
    "筒燈": ["崁燈", "嵌燈"],
    "崁燈": ["筒燈", "嵌燈"],
    "T-BAR": ["輕鋼架", "格柵"],
    "平板燈": ["面板燈", "輕鋼架燈"],
    "支架燈": ["層板燈"],
    "工事燈": ["工礦燈", "作業燈"],
}

CATEGORY_KEYWORDS = [
    ("平板燈", ["平板燈", "面板燈"]), ("投射燈", ["投射燈", "投光燈"]),
    ("泛光燈", ["泛光燈"]), ("軌道燈", ["軌道燈", "軌道投射"]),
    ("崁燈", ["崁燈", "嵌燈"]), ("筒燈", ["筒燈"]),
    ("支架燈", ["支架燈"]), ("工事燈", ["工事燈"]),
    ("吸頂燈", ["吸頂燈"]), ("吊燈", ["吊燈"]), ("壁燈", ["壁燈"]),
    ("路燈", ["路燈"]), ("高燈", ["高燈", "景觀燈"]),
    ("步道燈", ["步道燈", "車道燈", "引導燈"]),
    ("防潮燈", ["防潮燈"]), ("日光燈", ["日光燈"]),
    ("洗牆燈", ["洗牆燈"]), ("輕鋼架燈", ["輕鋼架"]),
    ("燈管", ["燈管"]), ("燈泡", ["燈泡"]),
]

CAT_MAP = {
    "吊管式吊燈": "工事燈", "嵌入式日光燈": "輕鋼架燈", "防水吸頂燈": "防潮燈",
    "T8支架燈": "支架燈", "夜間指引燈": "步道燈", "LED面板": "平板燈",
    "T-BAR燈": "輕鋼架燈",
    "泛光燈": "泛光燈", "投射燈": "泛光燈", "探照燈": "泛光燈", "聚光燈": "泛光燈",
    "輕鋼架燈": "輕鋼架燈", "格柵燈": "輕鋼架燈", "辦公室燈": "輕鋼架燈",
    "工事燈": "工事燈", "吊管燈": "工事燈", "停車場燈": "工事燈",
    "中東燈": "中東燈",
    "崁燈": "崁燈", "筒燈": "崁燈", "嵌燈": "崁燈", "下照燈": "崁燈", "downlight": "崁燈",
    "步道燈": "步道燈", "車道燈": "步道燈", "階梯燈": "步道燈", "引導燈": "步道燈",
    "吸頂燈": "吸頂燈", "環形燈": "吸頂燈", "雲朵燈": "吸頂燈",
    "平板燈": "平板燈", "面板燈": "平板燈",
    "支架燈": "支架燈", "層板燈": "支架燈",
    "防潮燈": "防潮燈", "浴室燈": "防潮燈",
    "軌道燈": "軌道燈", "吊燈": "吊燈", "壁燈": "壁燈",
    "路燈": "路燈", "高燈": "高燈", "景觀燈": "高燈",
    "洗牆燈": "洗牆燈", "燈管": "燈管", "燈泡": "燈泡",
}

CATEGORY_SYN_MAP = {
    "投射燈": {"泛光燈", "投射燈", "探照燈"},
    "泛光燈": {"投射燈", "泛光燈"},
    "崁燈": {"崁燈", "筒燈"}, "筒燈": {"崁燈", "筒燈"},
    "步道燈": {"步道燈", "車道燈"}, "車道燈": {"步道燈", "車道燈"},
    "平板燈": {"平板燈"}, "日光燈": {"日光燈", "支架燈"},
}


def _extract_products(page_num, text, page_img_descs):
    model_pattern = r'[A-Z][A-Z0-9-]{3,}[A-Z0-9](?:\b|$)'
    all_models = list(dict.fromkeys(re.findall(model_pattern, text)))
    if not all_models:
        return []

    lines = text.split("\n")
    product_names = []
    for line in lines:
        line = line.strip()
        if "燈" in line and 3 < len(line) < 25:
            if not any(kw in line for kw in ["產品型號", "消耗", "輸入", "材質", "色溫", "光通量", "型錄"]):
                product_names.append(line)

    series_name = product_names[0] if product_names else f"第{page_num}頁產品"
    wattages = list(dict.fromkeys(re.findall(r'(\d+)W\b', text)))
    color_temps = list(dict.fromkeys(re.findall(r'(\d+)K\b', text)))
    lumens = list(dict.fromkeys(re.findall(r'(\d+)\s*LM\b', text, re.I)))
    ip_match = re.findall(r'IP\s*(\d+)', text)
    ip_rating = ip_match[0] if ip_match else ""

    name_text = series_name + " " + " ".join(product_names)
    category = ""
    for cat, kws in CATEGORY_KEYWORDS:
        if any(kw in name_text for kw in kws):
            category = cat
            break
    if not category:
        for cat, kws in CATEGORY_KEYWORDS:
            if any(kw in text for kw in kws):
                category = cat
                break

    features = [f for f in ["節能標章", "感應", "調光", "防眩", "防水", "全電壓"] if f in text]

    meta = {"page": page_num, "series_name": series_name, "category": category,
            "models": ",".join(all_models[:10]), "wattages": ",".join(wattages[:5]),
            "color_temps": ",".join(color_temps[:5]), "lumens": ",".join(lumens[:5]),
            "ip_rating": ip_rating, "features": ",".join(features)}

    parts = [
        f"【產品】{series_name}", f"【類別】{category}" if category else "",
        f"【型號】{', '.join(all_models[:8])}", f"【瓦數】{', '.join(wattages[:5])}W" if wattages else "",
        f"【色溫】{', '.join(color_temps[:5])}K" if color_temps else "",
        f"【光通量】{', '.join(lumens[:5])}LM" if lumens else "",
        f"【IP防護】IP{ip_rating}" if ip_rating else "",
        f"【特色】{', '.join(features)}" if features else "",
    ]
    summary = "\n".join(p for p in parts if p)
    img_desc = "\n" + "\n".join(page_img_descs.get(page_num, []))
    chunk_text = f"{summary}\n\n--- 原始內容 (第{page_num}頁) ---\n{text}{img_desc}"
    return [{"text": chunk_text, "metadata": meta}]


def _add_synonyms(text):
    extra = set()
    for kw, syns in SYNONYMS.items():
        if kw in text:
            extra.update(syns)
    return text + " " + " ".join(extra) if extra else text


def _decompose_query(query):
    specs = {}
    for term, cat in CAT_MAP.items():
        if term in query:
            specs["category"] = cat
            break
    m = re.search(r'[≦<]\s*(\d+)\s*W', query)
    if m:
        specs["max_wattage"] = int(m.group(1))
    else:
        m = re.search(r'(\d+)\s*W.*以下', query)
        if m:
            specs["max_wattage"] = int(m.group(1))
    m = re.search(r'(\d{3,5})\s*K', query)
    if m:
        specs["color_temp"] = int(m.group(1))
    m = re.search(r'[≧>]\s*(\d+)\s*(?:lm|LM)', query, re.I)
    if m:
        specs["min_lumens"] = int(m.group(1))
    m = re.search(r'IP\s*(\d+)', query)
    if m:
        specs["ip_rating"] = int(m.group(1))
    return specs


def _metadata_filter(specs, metas):
    if not specs:
        return set(range(len(metas)))
    valid = set(range(len(metas)))
    cat = specs.get("category", "")
    if cat:
        cat_syns = CATEGORY_SYN_MAP.get(cat, {cat})
        cat_indices = set()
        for i, m in enumerate(metas):
            mc = m.get("category", "")
            if mc in cat_syns or mc == cat or (cat and cat in mc):
                cat_indices.add(i)
        if cat_indices:
            valid &= cat_indices
    max_w = specs.get("max_wattage")
    if max_w and isinstance(max_w, (int, float)):
        w_idx = set()
        for i, m in enumerate(metas):
            ws = m.get("wattages", "")
            if not ws:
                w_idx.add(i); continue
            try:
                if any(int(w) <= max_w * 1.2 for w in ws.split(",") if w.strip()):
                    w_idx.add(i)
            except Exception:
                w_idx.add(i)
        valid &= w_idx
    return valid


def _expand_specs(text):
    t = text
    t = re.sub(r'(\d+)\s*lm/W', r'\1發光效率(lm/W)', t)
    t = re.sub(r'([A-Za-z])(\d)', r'\1 \2', t)
    t = re.sub(r'(\d+)\s*W(?![a-zA-Z/])', lambda m: m.group(1) + "瓦(W)", t)

    def _ek(m):
        if t[:m.start()].endswith("色溫"):
            return m.group(1) + "K"
        return m.group(1) + "色溫(K)"
    t = re.sub(r'(\d+)\s*K(?![a-zA-Z])', _ek, t)
    t = re.sub(r'(\d+)\s*lm(?![/a-zA-Z])', lambda m: m.group(1) + "光通量(lm)", t, flags=re.I)
    t = re.sub(r'(\d+)\s*V(?![a-zA-Z])', lambda m: m.group(1) + "電壓(V)", t)
    t = t.replace("≦", "小於等於").replace("≧", "大於等於")
    return t


def _get_query_embedder():
    global _query_embed_model
    if _query_embed_model is None:
        print("[rag_engine] Loading BGE-M3 embedder for queries...")
        _query_embed_model = SentenceTransformer(EMBED_MODEL)
    return _query_embed_model


def _embed_query(query):
    model = _get_query_embedder()
    emb = model.encode([query[:4000]], normalize_embeddings=True)[0]
    return emb.astype(np.float32)


def _hybrid_retrieve(query, bm25_query, top_k, valid_indices):
    bm25 = _state["bm25"]
    chunks = _state["chunks"]
    metas = _state["chunk_metas"]
    embs = _state["chunk_embeddings"]

    qt = list(jieba.cut(bm25_query))
    bm25_scores = bm25.get_scores(qt)
    bm25_max = bm25_scores.max() if bm25_scores.max() > 0 else 1
    bm25_norm = bm25_scores / bm25_max

    q_emb = _embed_query(query)
    cos_scores = (embs @ q_emb).flatten()
    vec_min, vec_max = cos_scores.min(), cos_scores.max()
    vec_norm = (cos_scores - vec_min) / (vec_max - vec_min + 1e-8)

    hybrid = BM25_WEIGHT * bm25_norm + VECTOR_WEIGHT * vec_norm
    if valid_indices is not None and len(valid_indices) < len(chunks):
        for i in range(len(chunks)):
            hybrid[i] *= 1.3 if i in valid_indices else 0.7

    top_idx = np.argsort(hybrid)[::-1][:top_k]
    return [{"text": chunks[i], "metadata": metas[i], "score": float(hybrid[i]),
             "bm25_score": float(bm25_norm[i]), "vector_score": float(vec_norm[i])}
            for i in top_idx]


def _get_reranker():
    global _reranker
    if _reranker is None:
        print(f"[rag_engine] Loading BGE-reranker ({RERANK_MODEL})...")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def _bge_rerank(query, candidates, top_k):
    """Cross-encoder rerank using BAAI/bge-reranker-v2-m3."""
    shortlist = candidates[:RERANK_CANDIDATES]
    if not shortlist:
        return []

    reranker = _get_reranker()
    pairs = [(query, c["text"][:2000]) for c in shortlist]
    try:
        scores = reranker.predict(pairs).tolist()
    except Exception as e:
        print(f"[BGE Rerank] failed, falling back to hybrid score: {e}")
        for c in candidates:
            c["rerank_score"] = c["score"]
            c["rerank_reason"] = "(rerank 失敗)"
        candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        return candidates[:top_k]

    for c, s in zip(shortlist, scores):
        c["rerank_score"] = float(s)
        c["rerank_reason"] = ""

    for c in candidates[RERANK_CANDIDATES:]:
        c["rerank_score"] = float(c["score"])
        c["rerank_reason"] = "(未進入 rerank)"

    candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return candidates[:top_k]


def initialize():
    """Build chunks, BM25 index, and load the cached embeddings. Idempotent."""
    if _state:
        return

    print("[rag_engine] Loading PDF text via PyMuPDF...")
    doc = fitz.open(PDF_PATH)
    pymupdf_texts = {}
    for i in range(len(doc)):
        t = doc[i].get_text().strip()
        if t:
            pymupdf_texts[i + 1] = t
    doc.close()

    print("[rag_engine] Loading opendataloader image metadata...")
    with open(ODL_JSON, "r") as f:
        odl_data = json.load(f)
    odl_images = {}
    for kid in odl_data["kids"]:
        pn = kid.get("page number", 0)
        if kid.get("type") == "image":
            src = kid.get("source", "")
            if src:
                fp = os.path.join(ODL_DIR, src)
                if os.path.exists(fp) and os.path.getsize(fp) >= MIN_IMG_SIZE:
                    odl_images.setdefault(pn, []).append(fp)

    with open(IMG_CACHE_FILE, "r") as f:
        img_cache = json.load(f)
    page_img_descs = {}
    for pn, fps in odl_images.items():
        descs = [f"[圖片{os.path.basename(fp)}] {img_cache[fp]}"
                 for fp in fps if fp in img_cache]
        if descs:
            page_img_descs[pn] = descs

    print("[rag_engine] Chunking...")
    chunks, metas = [], []
    for pn in sorted(pymupdf_texts.keys()):
        text = pymupdf_texts[pn]
        prods = _extract_products(pn, text, page_img_descs)
        if prods:
            for p in prods:
                chunks.append(p["text"])
                metas.append(p["metadata"])
        else:
            chunks.append(f"[第{pn}頁]\n{text[:1000]}")
            metas.append({"page": pn, "series_name": "", "category": "",
                          "models": "", "wattages": "", "color_temps": "",
                          "lumens": "", "ip_rating": "", "features": ""})

    print(f"[rag_engine] {len(chunks)} chunks")

    for term in LAMP_TERMS:
        jieba.add_word(term)
    tokenized = [[t.strip() for t in jieba.cut(c) if t.strip()] for c in chunks]
    bm25 = BM25Okapi(tokenized)

    if not os.path.exists(EMBED_CACHE):
        raise RuntimeError(f"Embedding cache missing: {EMBED_CACHE}. "
                           "Run the v7 pipeline (dancelight_rag2.0_2026-04-29/"
                           "dancelight_rag_test.py) once to build it.")
    embs = np.load(EMBED_CACHE)
    if embs.shape[0] != len(chunks):
        raise RuntimeError(f"Embedding cache has {embs.shape[0]} rows but "
                           f"chunker produced {len(chunks)} chunks. "
                           "Delete the cache and rebuild.")

    _state.update({"chunks": chunks, "chunk_metas": metas,
                   "bm25": bm25, "chunk_embeddings": embs})
    print(f"[rag_engine] ready — embeddings {embs.shape}")


def search(query: str, top_k: int = 5):
    """Retrieve + qwen rerank. Returns a list of dicts shaped for the web UI."""
    if not _state:
        initialize()
    specs = _decompose_query(query)
    valid = _metadata_filter(specs, _state["chunk_metas"])
    expanded = _expand_specs(query)
    bm25_q = _add_synonyms(query)
    candidates = _hybrid_retrieve(expanded, bm25_q, RETRIEVE_K, valid)
    reranked = _bge_rerank(query, candidates, top_k=top_k)

    results = []
    for c in reranked:
        m = c["metadata"]
        page = m.get("page", 0)
        results.append({
            "page": page,
            "name": m.get("series_name") or f"第{page}頁產品",
            "models": m.get("models", ""),
            "category": m.get("category", ""),
            "wattages": m.get("wattages", ""),
            "color_temps": m.get("color_temps", ""),
            "lumens": m.get("lumens", ""),
            "ip_rating": m.get("ip_rating", ""),
            "features": m.get("features", ""),
            "score": round(float(c.get("rerank_score", 0)), 2),
            "reason": c.get("rerank_reason", ""),
        })
    return results
