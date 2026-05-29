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
from flask import Flask, g, jsonify, render_template, request, send_file

from rag import engine as rag_engine

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
    con.commit()
    con.close()


print("[web] Initializing RAG engine...")
rag_engine.initialize()
print("[web] Opening PDF for page rendering...")
_pdf_doc = fitz.open(rag_engine.PDF_PATH)
_page_cache: dict = {}
print(f"[web] Initializing SQLite at {DB_PATH}...")
init_db()


@app.get("/")
def index():
    return render_template("index.html", llm=rag_engine.LOCAL_LLM)


@app.post("/api/search")
def api_search():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    t0 = time.time()
    try:
        results = rag_engine.search(query, top_k=5)
    except Exception as e:
        return jsonify({"error": f"search failed: {e}"}), 500
    elapsed_ms = int((time.time() - t0) * 1000)

    query_id = None
    try:
        cur = _db().execute(
            "INSERT INTO queries (ts, query, llm_model, elapsed_ms, picks_json, client_ip, user_agent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                query,
                rag_engine.LLM_SELECT_MODEL,
                elapsed_ms,
                json.dumps(results, ensure_ascii=False),
                request.headers.get("CF-Connecting-IP") or request.remote_addr or "",
                (request.headers.get("User-Agent") or "")[:300],
            ),
        )
        _db().commit()
        query_id = cur.lastrowid
    except Exception as e:
        print(f"[db] insert failed: {e}")

    return jsonify({"query": query, "query_id": query_id,
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


@app.get("/api/page_image/<int:page>.png")
def api_page_image(page: int):
    if page < 1 or page > len(_pdf_doc):
        return "page not found", 404
    try:
        dpi = int(request.args.get("dpi", "110"))
    except ValueError:
        dpi = 110
    dpi = max(50, min(dpi, 300))
    key = (page, dpi)
    if key not in _page_cache:
        pix = _pdf_doc[page - 1].get_pixmap(dpi=dpi)
        _page_cache[key] = pix.tobytes("png")
    return send_file(io.BytesIO(_page_cache[key]), mimetype="image/png")


@app.get("/api/pdf_meta")
def api_pdf_meta():
    return jsonify({"total_pages": len(_pdf_doc)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
