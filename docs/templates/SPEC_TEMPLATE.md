# Spec: <feature-name>

**Status**: draft | approved | in-progress | done
**Author**: <name>  **Date**: YYYY-MM-DD
**Affected modules**: rag / web / research (list)

---

## 目標 (Goal)

One paragraph in plain language. What user-visible thing changes and why?
A reader who has never seen this codebase should understand the motivation.

---

## 約束 (Constraints)

What we may NOT break:

- [ ] Other modules' CONTRACT.md (list which ones this touches, if any)
- [ ] Performance budget (e.g. /api/search p95 < 2s)
- [ ] Cost budget (e.g. ≤ 1 GPT-4o call per query)
- [ ] Compatibility with existing `dancelight_queries.db` schema
- [ ] Embedding cache invariant (chunk_text md5 keys)
- [ ] Public URL behavior
- [ ] (other domain-specific constraints)

Anything that requires changing another module's contract must be called out
here explicitly. That contract change must land in the SAME pull-request and
its CONTRACT.md must be edited as part of this work.

---

## 成功條件 (Success criteria)

Each one must be a verifiable check, ideally a test name:

- [ ] `tests/contract/test_<module>_contract.py::test_<name>` passes
- [ ] Manual: `curl ...` returns ...
- [ ] User can do <X> via the UI and see <Y>

If any criterion can't be expressed as a test, say why.

---

## Non-goals

What we are explicitly NOT solving in this spec. Prevents scope creep.

---

## Open questions

(Resolve before user approves.)

---

## Implementation sketch

Optional. High-level only — the spec is about WHAT and WHY, not HOW.
