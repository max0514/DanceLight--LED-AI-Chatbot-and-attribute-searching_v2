"""Contract tests for web/ — see web/CONTRACT.md.

Lightweight tests use Flask's test client without touching `rag.engine`.
The @pytest.mark.heavy test runs a real query end-to-end.

We mock `rag.engine.search` for the lightweight tests so we can verify route
shape without loading models. Public constants on `rag.engine` are also
required to import `web.app`, so we don't mock those.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    """Flask test client with rag.engine.search mocked and a temp SQLite."""
    # Use a temp DB so tests don't pollute production data.
    monkeypatch.chdir(tmp_path)

    # Import after chdir so DB_PATH resolves under tmp_path.
    # rag.engine.initialize() is still called at module import; we patch it
    # to a no-op to avoid loading models.
    with patch("rag.engine.initialize", return_value=None), \
         patch("rag.engine.search", return_value=_fake_results()):
        # fitz.open() at module level needs a real PDF path; use a stub.
        with patch("fitz.open") as mock_open:
            mock_open.return_value = _FakePdfDoc(total_pages=10)
            # Reload web.app to re-trigger module init under the patched env
            import importlib
            import web.app as web_app
            importlib.reload(web_app)
            yield web_app.app.test_client()


def _fake_results():
    return [
        {
            "rank_label": "★ 推薦",
            "name": "微笑崁燈系列",
            "category": "崁燈",
            "page": 115,
            "models": "LED-9DOS15DR3",
            "wattages": "15",
            "color_temps": "6000,4000,3000",
            "lumens": "1500",
            "ip_rating": "",
            "features": "全電壓",
            "score": 0.95,
            "reason": "15W崁燈，色溫6000K接近需求",
        },
    ] + [
        {
            "rank_label": f"備選 {i}",
            "name": f"備選產品{i}",
            "category": "崁燈",
            "page": 100 + i,
            "models": f"MODEL-{i}",
            "wattages": "15",
            "color_temps": "6000",
            "lumens": "1500",
            "ip_rating": "",
            "features": "",
            "score": 0.8 - 0.1 * i,
            "reason": f"備選 {i} 理由",
        }
        for i in range(1, 5)
    ]


class _FakePdfDoc:
    def __init__(self, total_pages):
        self._n = total_pages

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return self  # support _pdf_doc[page-1].get_pixmap()

    def get_pixmap(self, dpi):
        # Return a tiny PNG-bytes-producing stub
        class _Pix:
            def tobytes(self, fmt):
                # Minimal 1x1 PNG
                return (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc\xf8\xff"
                        b"\xff?\x00\x05\xfe\x02\xfe\xa3\xc6\xc6\xf6\x00\x00\x00\x00IEND\xaeB`\x82")
        return _Pix()


# --- /api/search ---

def test_search_empty_query_returns_400(client):
    r = client.post("/api/search", json={"query": ""})
    assert r.status_code == 400
    assert "error" in r.get_json()


def test_search_missing_query_returns_400(client):
    r = client.post("/api/search", json={})
    assert r.status_code == 400


def test_search_returns_documented_shape(client):
    r = client.post("/api/search", json={"query": "15W崁燈"})
    assert r.status_code == 200
    body = r.get_json()
    assert set(body.keys()) >= {"query", "query_id", "elapsed_ms", "results"}
    assert body["query"] == "15W崁燈"
    assert isinstance(body["elapsed_ms"], int)
    assert isinstance(body["results"], list)
    assert len(body["results"]) == 5
    assert body["results"][0]["rank_label"] == "★ 推薦"


def test_search_writes_to_sqlite(client, tmp_path):
    r = client.post("/api/search", json={"query": "test"})
    body = r.get_json()
    assert body["query_id"] is not None and isinstance(body["query_id"], int)

    con = sqlite3.connect(tmp_path / "dancelight_queries.db")
    row = con.execute(
        "SELECT id, query, picks_json FROM queries WHERE id=?",
        (body["query_id"],),
    ).fetchone()
    con.close()
    assert row is not None
    assert row[1] == "test"
    picks = json.loads(row[2])
    assert len(picks) == 5


# --- /api/feedback ---

def test_feedback_requires_query_id_int(client):
    r = client.post("/api/feedback", json={"kind": "click"})
    assert r.status_code == 400


def test_feedback_requires_kind(client):
    r = client.post("/api/feedback", json={"query_id": 1, "kind": ""})
    assert r.status_code == 400


def test_feedback_happy_path(client):
    # Need an existing query_id first
    s = client.post("/api/search", json={"query": "x"})
    qid = s.get_json()["query_id"]
    r = client.post("/api/feedback", json={"query_id": qid, "kind": "click", "payload": {"i": 1}})
    assert r.status_code == 200
    assert r.get_json() == {"ok": True}


# --- /api/page_image ---

def test_page_image_out_of_range_404(client):
    r = client.get("/api/page_image/9999.png")
    assert r.status_code == 404


def test_page_image_default_dpi(client):
    r = client.get("/api/page_image/1.png")
    assert r.status_code == 200
    assert r.content_type == "image/png"


def test_page_image_dpi_clamp(client):
    # dpi=10000 should be clamped to 300
    r = client.get("/api/page_image/1.png?dpi=10000")
    assert r.status_code == 200


# --- /api/pdf_meta ---

def test_pdf_meta_returns_total_pages(client):
    r = client.get("/api/pdf_meta")
    assert r.status_code == 200
    body = r.get_json()
    assert "total_pages" in body
    assert isinstance(body["total_pages"], int)
    assert body["total_pages"] == 10  # fake doc
