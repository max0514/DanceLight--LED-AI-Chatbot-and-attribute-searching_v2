# <module-name> — Summary

**Auto-maintained.** Edit this file after any non-trivial change to the module.
A PostToolUse hook reminds Claude Code to refresh it.

---

## What this module does

2–4 sentences a new contributor can read in 30 seconds. No jargon. Like
explaining to a teammate at lunch.

## Key files

| File | Purpose |
|---|---|
| `engine.py` | what it owns |
| ... | ... |

## How to use it (from outside)

```python
from <module> import ...
```

One short example showing the canonical usage pattern.

## Internal notes (gotchas)

Things that surprised the person who wrote this. E.g.:
- "Cache keys depend on chunk_text — never change chunking without rebuilding the cache."
- "Reranker uses 2000-char truncation; long docs lose tail context."

## Last updated

YYYY-MM-DD — what changed
