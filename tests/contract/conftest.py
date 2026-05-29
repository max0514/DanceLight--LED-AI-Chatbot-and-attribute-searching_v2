"""Shared fixtures for contract tests.

Contract tests verify the documented public interface of each module
(see <module>/CONTRACT.md). They should NOT depend on:
- network calls (mock OpenAI / Ollama)
- the real PDF being present (use a tiny fixture if needed)
- heavy embedding model loads (mock or skip in CI; allow in local)

If a test needs real models, mark it with @pytest.mark.heavy so it can
be skipped via `pytest -m "not heavy"`.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def pytest_collection_modifyitems(config, items):
    # Mark heavy tests so they're easy to filter
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(pytest.mark.heavy)
