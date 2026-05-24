"""Flask web UI for the dancelight_rag_api_test pipeline with qwen3.5:122b reranker.

Run:
    cd /var/lib/jenkins/dancelight_data && python3 dancelight_web_app.py
Then open http://<host>:8000/
"""
import io

import fitz
from flask import Flask, jsonify, render_template, request, send_file

import rag_engine

app = Flask(__name__)

print("[web] Initializing RAG engine...")
rag_engine.initialize()
print("[web] Opening PDF for page rendering...")
_pdf_doc = fitz.open(rag_engine.PDF_PATH)
_page_cache: dict = {}


@app.get("/")
def index():
    return render_template("index.html", llm=rag_engine.LOCAL_LLM)


@app.post("/api/search")
def api_search():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    if not query:
        return jsonify({"error": "empty query"}), 400
    try:
        results = rag_engine.search(query, top_k=5)
    except Exception as e:
        return jsonify({"error": f"search failed: {e}"}), 500
    return jsonify({"query": query, "results": results})


@app.get("/api/page_image/<int:page>.png")
def api_page_image(page: int):
    # Metadata pages are 1-indexed; PyMuPDF is 0-indexed.
    if page < 1 or page > len(_pdf_doc):
        return "page not found", 404
    if page not in _page_cache:
        pix = _pdf_doc[page - 1].get_pixmap(dpi=110)
        _page_cache[page] = pix.tobytes("png")
    return send_file(io.BytesIO(_page_cache[page]), mimetype="image/png")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
