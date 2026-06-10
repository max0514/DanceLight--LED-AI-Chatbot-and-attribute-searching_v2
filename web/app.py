"""Flask web UI for dancelight RAG: BGE hybrid retrieve + GPT-4o LLM selection.

Run:
    cd /root/dancelight-rag-repo && python3 -m web.app
Then open http://<host>:8000/

Queries and LLM picks are logged to ./dancelight_queries.db (SQLite).
"""
import io
import json
import sqlite3
import time
from datetime import datetime, timezone

import fitz
from dotenv import load_dotenv
from flask import Flask, g, jsonify, render_template, request, send_file

load_dotenv(".env")

from rag import engine as rag_engine
from web import line_handler as line_module

app = Flask(__name__)

DB_PATH = "./dancelight_queries.db"


def _db():
    db = getattr(g, "_db", None)
    if db is None:
        db = g._db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
    return db


@app.teardown_appcontext
def _close_db(_exc):
    db = getattr(g, "_db", None)
    if db is not None:
        db.close()


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            query TEXT NOT NULL,
            llm_model TEXT,
            elapsed_ms INTEGER,
            picks_json TEXT NOT NULL,
            client_ip TEXT,
            user_agent TEXT
        );
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id INTEGER NOT NULL,
            ts TEXT NOT NULL,
            kind TEXT NOT NULL,
            payload TEXT,
            FOREIGN KEY (query_id) REFERENCES queries(id)
        );
        CREATE INDEX IF NOT EXISTS idx_queries_ts ON queries(ts);
        CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback(query_id);
        """
    )
    # Idempotent ALTER for the dual-edition rollout.
    cols = {row[1] for row in con.execute("PRAGMA table_info(queries)").fetchall()}
    if "edition" not in cols:
        con.execute("ALTER TABLE queries ADD COLUMN edition TEXT")
    con.commit()
    con.close()


print("[web] Initializing RAG engine (all editions)...")
rag_engine.initialize()      # loads every edition in CORPUS
print("[web] Opening PDFs for page rendering...")
_pdf_docs: dict[str, fitz.Document] = {
    ed: fitz.open(cfg["pdf"]) for ed, cfg in rag_engine.CORPUS.items()
}
_page_cache: dict = {}        # keyed by (edition, page, dpi)
print(f"[web] Initializing SQLite at {DB_PATH}...")
init_db()
line_module.register(app)


def _resolve_edition(value) -> str:
    """Validate caller-supplied edition; fall back to DEFAULT_EDITION."""
    if isinstance(value, str) and value in rag_engine.CORPUS:
        return value
    return rag_engine.DEFAULT_EDITION


@app.get("/")
def index():
    editions = [{"key": k, "label": v["label"]} for k, v in rag_engine.CORPUS.items()]
    return render_template("index.html",
                           llm=rag_engine.LOCAL_LLM,
                           editions=editions,
                           default_edition=rag_engine.DEFAULT_EDITION)


@app.get("/api/editions")
def api_editions():
    return jsonify({
        "default": rag_engine.DEFAULT_EDITION,
        "editions": [{"key": k, "label": v["label"]}
                     for k, v in rag_engine.CORPUS.items()],
    })


@app.post("/api/search")
def api_search():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    edition = _resolve_edition(data.get("edition"))
    t0 = time.time()
    try:
        results = rag_engine.search(query, top_k=5, edition=edition)
    except Exception as e:
        return jsonify({"error": f"search failed: {e}"}), 500
    elapsed_ms = int((time.time() - t0) * 1000)

    query_id = None
    try:
        cur = _db().execute(
            "INSERT INTO queries (ts, query, llm_model, elapsed_ms, picks_json, "
            "client_ip, user_agent, edition) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                query,
                rag_engine.LLM_SELECT_MODEL,
                elapsed_ms,
                json.dumps(results, ensure_ascii=False),
                request.headers.get("CF-Connecting-IP") or request.remote_addr or "",
                (request.headers.get("User-Agent") or "")[:300],
                edition,
            ),
        )
        _db().commit()
        query_id = cur.lastrowid
    except Exception as e:
        print(f"[db] insert failed: {e}")

    return jsonify({"query": query, "query_id": query_id, "edition": edition,
                    "elapsed_ms": elapsed_ms, "results": results})


@app.post("/api/feedback")
def api_feedback():
    data = request.get_json(silent=True) or {}
    qid = data.get("query_id")
    kind = (data.get("kind") or "").strip()
    if not isinstance(qid, int) or not kind:
        return jsonify({"error": "query_id (int) and kind required"}), 400
    payload = data.get("payload")
    try:
        _db().execute(
            "INSERT INTO feedback (query_id, ts, kind, payload) VALUES (?, ?, ?, ?)",
            (
                qid,
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                kind,
                json.dumps(payload, ensure_ascii=False) if payload is not None else None,
            ),
        )
        _db().commit()
    except Exception as e:
        return jsonify({"error": f"feedback insert failed: {e}"}), 500
    return jsonify({"ok": True})


def _serve_page_image(edition: str, page: int):
    doc = _pdf_docs.get(edition)
    if doc is None or page < 1 or page > len(doc):
        return "page not found", 404
    try:
        dpi = int(request.args.get("dpi", "110"))
    except ValueError:
        dpi = 110
    dpi = max(50, min(dpi, 300))
    key = (edition, page, dpi)
    if key not in _page_cache:
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        _page_cache[key] = pix.tobytes("png")
    return send_file(io.BytesIO(_page_cache[key]), mimetype="image/png")


@app.get("/api/page_image/<edition>/<int:page>.png")
def api_page_image_edition(edition: str, page: int):
    return _serve_page_image(_resolve_edition(edition), page)


@app.get("/api/page_image/<int:page>.png")
def api_page_image(page: int):
    """Backward-compat: defaults to DEFAULT_EDITION."""
    return _serve_page_image(rag_engine.DEFAULT_EDITION, page)


@app.get("/api/pdf_meta")
def api_pdf_meta():
    edition = _resolve_edition(request.args.get("edition"))
    return jsonify({"edition": edition, "total_pages": len(_pdf_docs[edition])})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
