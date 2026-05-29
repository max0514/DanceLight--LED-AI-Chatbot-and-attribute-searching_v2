# Contract: <module-name>

This file documents the **stable external interface** of this module.
Callers may rely on everything documented here. Anything NOT documented here
is implementation detail and may change without notice.

Changes to this file require a corresponding spec.md in `/specs/`.

---

## Public API

### `function_name(arg1: T, arg2: T = default) -> ReturnT`

What it does (one line).

**Inputs**:
- `arg1` — meaning, valid range
- `arg2` — meaning, default behavior

**Returns**: shape and meaning

**Raises**: only document exceptions callers are expected to catch

**Side effects**: writes to file X, reads env var Y, makes HTTP call to Z

---

## Public constants

- `CONSTANT_NAME` (type) — meaning, when it may change

---

## Invariants

Properties of the module that callers may rely on:

1. `initialize()` is idempotent and must be called before any other function.
2. ...

---

## Backwards compatibility

What changes are considered BREAKING (require major version bump in spec /
all-callers update):

- Removing a function or constant
- Renaming a parameter
- Changing return shape
- Changing exception types raised
- ...

What changes are NON-BREAKING (safe in minor changes):

- Adding new optional parameters
- Adding new fields to return dicts
- Internal refactors
- ...
