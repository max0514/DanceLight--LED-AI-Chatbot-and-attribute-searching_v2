#!/usr/bin/env python3
"""PostToolUse hook: remind Claude Code to refresh module SUMMARY.md.

Claude Code calls this after every Edit/Write. It reads the hook input on
stdin (JSON), figures out if a module file was touched, and emits a JSON
payload that injects a reminder into the conversation context.

Idempotent / fast / silent when no module touched.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path("/root/dancelight-rag-repo")
MODULES = ("rag", "web", "research")


def module_for(path: Path) -> str | None:
    """Return module name if path lives under a tracked module package."""
    try:
        rel = path.resolve().relative_to(REPO_ROOT)
    except (ValueError, OSError):
        return None
    parts = rel.parts
    if not parts:
        return None
    head = parts[0]
    if head in MODULES:
        # Don't trigger when SUMMARY.md / CONTRACT.md themselves were edited
        if len(parts) >= 2 and parts[-1] in ("SUMMARY.md", "CONTRACT.md"):
            return None
        return head
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # nothing to do

    tool_input = payload.get("tool_input") or {}
    file_path = tool_input.get("file_path")
    if not file_path:
        return 0

    mod = module_for(Path(file_path))
    if not mod:
        return 0

    summary_path = REPO_ROOT / mod / "SUMMARY.md"
    rel_summary = summary_path.relative_to(REPO_ROOT)
    rel_edited = Path(file_path).resolve().relative_to(REPO_ROOT)

    msg = (
        f"You just edited `{rel_edited}` in module `{mod}/`. "
        f"Per CLAUDE.md §4.4, refresh `{rel_summary}` before ending the turn "
        f"if this change affects what the module does, how it's used, or any gotcha. "
        f"If the edit was trivial (typo, comment, log line), you may skip the summary update — "
        f"but say so explicitly in your reply."
    )

    out = {
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": msg,
        }
    }
    json.dump(out, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
