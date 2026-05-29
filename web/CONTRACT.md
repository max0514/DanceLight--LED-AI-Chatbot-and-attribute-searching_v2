# Contract: `web/`

Flask app exposing the HTTP API + UI. Public callers (browsers, scripts,
tests) may rely on every endpoint shape below. Internal Python symbols
(`_db`, `init_db`, `app`) are implementation detail.

Changes to this file require a corresponding `specs/<feature>.md`.

---

## HTTP API

Base URL (production): `https://dancelight-rag-nccu.loca.lt`
Base URL (dev): `http://127.0.0.1:8000`

### `GET /`

Returns the search UI (`templates/index.html`). The template is rendered with
context `{"llm": rag.engine.LOCAL_LLM}`. No query params.

---

### `POST /api/search`

Run a query through `rag.engine.search()` and log to SQLite.

**Request body** (JSON): `{"query": "<chinese text>"}`. Empty/missing query → 400.

**Response** (200 JSON):
```json
{
    "query": "<echo>",
    "query_id": <int|null>,          // SQLite row id; null on DB failure
    "elapsed_ms": <int>,             // search duration
    "results": [ ... ]               // list of 5 dicts per rag.engine.search() contract
}
```

**Errors**:
- 400 `{"error": "empty query"}` — empty/whitespace query
- 500 `{"error": "search failed: <msg>"}` — `rag.engine.search()` raised

**Side effects**: INSERTs one row into `queries` table.

---

### `POST /api/feedback`

Record user feedback on a prior search.

**Request body** (JSON):
```json
{
    "query_id": <int>,               // required, must match a queries.id
    "kind": "<string>",              // e.g. "click", "no_match"
    "payload": <any|null>            // optional, JSON-serializable
}
```

**Response** (200): `{"ok": true}`
**Errors**: 400 if `query_id` not int or `kind` empty; 500 on DB failure.

---

### `GET /api/page_image/<int:page>.png?dpi=<50-300>`

Render a single PDF page as PNG via PyMuPDF.

**Path param**: `page` — 1-indexed page number. Out of range → 404.
**Query param**: `dpi` — default 110, clamped to [50, 300]. Non-integer → fallback to 110.

**Response** (200): `Content-Type: image/png`. Cached per `(page, dpi)` in-memory.
**Errors**: 404 plain-text "page not found" if page out of range.

---

### `GET /api/pdf_meta`

Returns PDF metadata.

**Response** (200): `{"total_pages": <int>}`

---

## SQLite schema (load-bearing)

Tables created in `init_db()` at startup; changes to either schema require a
data-migration plan in the spec.

```sql
CREATE TABLE queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,             -- ISO 8601 UTC
    query       TEXT NOT NULL,
    llm_model   TEXT,
    elapsed_ms  INTEGER,
    picks_json  TEXT NOT NULL,             -- json.dumps(results)
    client_ip   TEXT,                       -- CF-Connecting-IP > remote_addr
    user_agent  TEXT                        -- truncated to 300 chars
);
CREATE TABLE feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id    INTEGER NOT NULL REFERENCES queries(id),
    ts          TEXT NOT NULL,
    kind        TEXT NOT NULL,
    payload     TEXT
);
```

**Public columns** (callers may read): all of the above.
**Schema invariants**: `queries.id` and `feedback.query_id` are stable; the
referential link is required for offline analysis.

---

## Entry point

Canonical run command (cron-managed in production):
```bash
cd /root/dancelight-rag-repo && python3 -m web.app
```

Port `8000` on all interfaces. Single-process (`threaded=True`).

---

## Invariants

1. `rag.engine.initialize()` runs once at startup; first request is fast.
2. SQLite tables exist before any request is served.
3. `/api/search` does NOT block on DB failure — search still returns; `query_id` is null.
4. Page images are cached forever in process memory; restart clears cache.
5. The Flask app reads `template_folder` from `web/templates/` (Flask default).

---

## Backwards compatibility

**BREAKING** (spec-gated):
- Adding required fields to any request body.
- Removing a response field.
- Changing an endpoint path or method.
- Schema-breaking DB migrations (DROP COLUMN, type change).

**NON-BREAKING**:
- Adding optional request fields.
- Adding new response fields.
- Adding new endpoints.
- Additive schema migrations (`ALTER TABLE ... ADD COLUMN`).
- UI/template changes that preserve API shape.

---

## Dependencies on other modules

`web/` depends on `rag.engine` (`initialize`, `search`, `PDF_PATH`,
`LOCAL_LLM`, `LLM_SELECT_MODEL`). It **must not** import from `research/`,
and **must not** reach into `rag.engine` private symbols (anything prefixed `_`).
