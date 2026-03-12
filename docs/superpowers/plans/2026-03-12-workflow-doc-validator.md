# Workflow Doc Validator Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `scripts/validate_features.py` from a feature-list shape check into a workflow/doc consistency gate that catches durable-state drift already observed in this repository.

**Architecture:** Keep the validator as the single entrypoint so existing workflows do not change. First, add focused Python unit tests that pin the new cross-file contract on synthetic repo fixtures. Then extend `scripts/validate_features.py` with conservative parsers for `task-progress.md`, verification-step denylist checks, and repo hygiene sentinels. Finish by running the validator against the real repo, fixing durable-state fallout, and recording the new feature in the long-task ledger.

**Tech Stack:** Python 3 stdlib (`json`, `pathlib`, `re`, `tempfile`, `unittest`), existing long-task durable files (`feature-list.json`, `task-progress.md`, `RELEASE_NOTES.md`), repository-local validator entrypoint `scripts/validate_features.py`

---

## Chunk 1: Lock The New Contract With Tests

### Task 1: Add focused validator tests on synthetic durable-state fixtures

**Files:**
- Create: `tests/test_validate_features.py`
- Modify: `scripts/validate_features.py`

- [x] **Step 1: Write a happy-path fixture test**

Create a temp repo fixture containing consistent `feature-list.json`, `task-progress.md`, and `RELEASE_NOTES.md`, then assert `validate(...)` returns no errors.

- [x] **Step 2: Write the first failing mismatch test**

Add a test where `task-progress.md` claims the wrong passing count and assert the validator reports a progress-summary mismatch.

- [x] **Step 3: Write the latest-session hygiene failure**

Add a test where the latest session block still contains `pending rerun after durable-state update` and assert the validator rejects it.

- [x] **Step 4: Write the verification denylist failure**

Add a test where a feature uses `cargo test --tests -q test_name` and assert the validator reports the bad command pattern.

- [x] **Step 5: Write the temp-artifact failure**

Add a test fixture with a tracked-looking `*.new` file and assert the validator reports the orphan artifact.

- [x] **Step 6: Run the validator tests to verify RED**

Run: `python3 -m unittest tests/test_validate_features.py`
Expected: FAIL because `scripts/validate_features.py` does not yet implement the cross-file checks.

## Chunk 2: Extend The Validator Minimally

### Task 2: Implement conservative workflow/doc checks in the existing script

**Files:**
- Modify: `scripts/validate_features.py`
- Test: `tests/test_validate_features.py`

- [x] **Step 1: Resolve companion durable files from the feature-list path**

Infer the repo root from the provided `feature-list.json` path and load `task-progress.md` and `RELEASE_NOTES.md` with explicit failure messages if missing.

- [x] **Step 2: Parse current focus, next feature, progress summary, and latest session**

Add narrow parsers that fail closed when the expected durable-state lines or latest session heading cannot be found.

- [x] **Step 3: Add workflow consistency checks**

Validate that current focus and next feature exist in the feature inventory, that the progress summary matches actual passing/failing counts, and that the latest session does not contain known stale placeholders.

- [x] **Step 4: Add verification-step denylist and repo hygiene sentinels**

Reject known bad verification commands and scan the repo for explicit temp-artifact suffixes such as `*.new`, `*.old`, and `*.tmp`.

- [x] **Step 5: Run the focused validator tests to verify GREEN**

Run: `python3 -m unittest tests/test_validate_features.py`
Expected: PASS

## Chunk 3: Prove The Real Repo Satisfies The New Gate

### Task 3: Integrate the new validator into the durable workflow honestly

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/superpowers/specs/2026-03-12-workflow-doc-validator-design.md`
- Modify: `docs/superpowers/plans/2026-03-12-workflow-doc-validator.md`

- [x] **Step 1: Add the validator feature to the inventory**

Add a new cross-cutting feature entry for the workflow/doc validator gate so the durable backlog reflects the user-directed scope change.

- [x] **Step 2: Run the validator on the real repo**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: FAIL only if it surfaces genuine repo drift that must be fixed before the feature can be closed.

- [x] **Step 3: Repair any real durable-state issues**

Fix current repo contradictions exposed by the stronger validator without weakening the checks.

- [x] **Step 4: Re-run verification for fresh evidence**

Run: `python3 -m unittest tests/test_validate_features.py`
Expected: PASS

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: PASS with updated passing/failing counts.

- [x] **Step 5: Persist feature closure**

Mark the new validator feature `passing`, update the current focus / next feature handoff in `task-progress.md`, and add a concise release note describing the new workflow guardrail.
