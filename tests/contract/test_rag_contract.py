"""Contract tests for rag/ — see rag/CONTRACT.md.

Lightweight tests verify the public API surface exists with the right shape.
The @pytest.mark.heavy tests load real models and run a real query — skip
in CI with `pytest -m 'not heavy'`.
"""
from __future__ import annotations

import inspect

import pytest


# --- lightweight contract checks (no model load) ---

def test_module_importable():
    from rag import engine  # noqa: F401


def test_initialize_exists_and_takes_no_args():
    from rag import engine
    assert callable(engine.initialize)
    sig = inspect.signature(engine.initialize)
    # No required parameters
    required = [p for p in sig.parameters.values()
                if p.default is inspect.Parameter.empty
                and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
    assert required == [], f"initialize() must take no required args; got {required}"


def test_search_signature():
    from rag import engine
    assert callable(engine.search)
    sig = inspect.signature(engine.search)
    params = sig.parameters
    assert "query" in params, "search() must accept 'query' parameter"
    assert "top_k" in params, "search() must accept 'top_k' parameter"
    assert params["top_k"].default == 5, "search() top_k default must be 5"


def test_public_constants_present():
    from rag import engine
    for name, expected_type in [
        ("LLM_SELECT_MODEL", str),
        ("LOCAL_LLM", str),
        ("PDF_PATH", str),
        ("EMBED_MODEL", str),
        ("RERANK_MODEL", str),
    ]:
        assert hasattr(engine, name), f"rag.engine must export {name}"
        assert isinstance(getattr(engine, name), expected_type), \
            f"rag.engine.{name} must be {expected_type.__name__}"


def test_local_llm_aliases_select_model():
    from rag import engine
    assert engine.LOCAL_LLM == engine.LLM_SELECT_MODEL, \
        "LOCAL_LLM is documented as alias for LLM_SELECT_MODEL"


# --- heavy contract checks (real model + real query) ---

REQUIRED_RESULT_KEYS = {
    "rank_label", "name", "category", "page", "models",
    "wattages", "color_temps", "lumens", "ip_rating", "features",
    "score", "reason",
}

VALID_RANK_LABELS = {"★ 推薦", "備選 1", "備選 2", "備選 3", "備選 4"}


@pytest.mark.heavy
def test_search_returns_documented_shape():
    """End-to-end: run a real query and assert the result list matches CONTRACT.md."""
    from rag import engine
    engine.initialize()

    results = engine.search("15W崁燈 6500K", top_k=5)

    assert isinstance(results, list), "search() must return a list"
    assert len(results) <= 5, f"search(top_k=5) returned {len(results)} > 5"
    assert len(results) >= 1, "search() returned empty — unexpected for this query"

    seen_labels = set()
    for i, r in enumerate(results):
        missing = REQUIRED_RESULT_KEYS - set(r.keys())
        assert not missing, f"result[{i}] missing keys: {missing}"
        assert r["rank_label"] in VALID_RANK_LABELS, \
            f"result[{i}] has unexpected rank_label: {r['rank_label']!r}"
        assert isinstance(r["page"], int) and r["page"] >= 1, \
            f"result[{i}].page must be 1-indexed int; got {r['page']!r}"
        assert isinstance(r["score"], (int, float)), \
            f"result[{i}].score must be numeric; got {type(r['score'])}"
        assert isinstance(r["name"], str) and r["name"].strip(), \
            f"result[{i}].name must be non-empty string"
        seen_labels.add(r["rank_label"])

    # The top pick must be ★ 推薦
    assert results[0]["rank_label"] == "★ 推薦", \
        f"results[0] must be the recommendation; got {results[0]['rank_label']!r}"


@pytest.mark.heavy
def test_name_not_a_bad_pattern():
    """Defense-in-depth: name must never be a warning line / marketing slogan / placeholder."""
    from rag import engine
    engine.initialize()
    results = engine.search("吸頂燈 24W 3000K", top_k=5)

    for i, r in enumerate(results):
        name = r["name"]
        assert not name.startswith(("※", "▲", "★", "●", "☆", "*", "▼")), \
            f"result[{i}].name has symbol prefix: {name!r}"
        assert "第" not in name or "頁產品" not in name, \
            f"result[{i}].name looks like placeholder: {name!r}"
        assert "營造" not in name and "氛圍" not in name, \
            f"result[{i}].name looks like marketing slogan: {name!r}"
