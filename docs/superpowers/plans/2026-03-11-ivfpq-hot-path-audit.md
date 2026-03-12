# IVF-PQ Hot Path Audit Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the current IVF/IVF-PQ ambiguity into an executable audit that clearly separates placeholder IVF scaffolding from the real IVF-PQ path eligible for parity and performance evaluation.

**Architecture:** Add a focused audit regression that proves the simplified `src/faiss/ivf.rs` code path is not the production/path-of-record IVF-PQ implementation, then capture the real hot path in `src/faiss/ivfpq.rs` with minimal doc/test evidence. Keep the change scoped to auditability rather than algorithm repair.

**Tech Stack:** Rust 2021, existing IVF/IVF-PQ tests, remote authority wrappers in `scripts/remote/test.sh`

---

## Chunk 1: Audit Red Test

### Task 1: Add a failing audit regression for the IVF-PQ path split

**Files:**
- Modify: `src/faiss/ivfpq.rs`
- Verify: `src/faiss/ivf.rs`

- [x] **Step 1: Write the failing test**

Add a unit test near the existing `IvfPqIndex` tests that proves the audited hot path owns residual PQ training / encoded inverted lists, while the standalone `src/faiss/ivf.rs` scaffold does not expose the same surfaces or semantics.

- [x] **Step 2: Run test to verify it fails**

Run: `cargo test ivfpq_hot_path_audit --lib -- --nocapture`
Expected: FAIL until the audit surface exists

- [x] **Step 3: Write minimal implementation**

Expose the smallest audit-only helper(s) or metadata needed to prove the hot-path split without changing IVF-PQ search behavior.

- [x] **Step 4: Run test to verify it passes**

Run: `cargo test ivfpq_hot_path_audit --lib -- --nocapture`
Expected: PASS

## Chunk 2: Durable Audit Narrative

### Task 2: Refresh the local audit/debug surfaces

**Files:**
- Modify: `tests/debug_ivf_pq_recall.rs`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`

- [x] **Step 1: Tighten the existing debug recall entrypoint**

Update `tests/debug_ivf_pq_recall.rs` so its printed narrative matches the audited path split and no longer implies that the simplified IVF scaffold is the measured hot path.

- [x] **Step 2: Update durable audit docs**

Record in `TASK_QUEUE.md` and `GAP_ANALYSIS.md` that `src/faiss/ivfpq.rs` is the real hot path under review and `src/faiss/ivf.rs` remains a placeholder/scaffold surface.

## Chunk 3: Authority Verification

### Task 3: Re-run the IVF/IVF-PQ authority checks

**Files:**
- Verify: `src/faiss/ivf.rs`
- Verify: `src/faiss/ivfpq.rs`
- Verify: `tests/debug_ivf_pq_recall.rs`

- [x] **Step 1: Run authority IVF library lane**

Run: `bash scripts/remote/test.sh --command "cargo test --lib ivf -- --nocapture"`
Expected: `test=ok`

- [x] **Step 2: Run authority IVF-PQ library lane**

Run: `bash scripts/remote/test.sh --command "cargo test --lib ivfpq -- --nocapture"`
Expected: `test=ok`

- [x] **Step 3: Run authority debug recall lane**

Run: `bash scripts/remote/test.sh --command "cargo test --test debug_ivf_pq_recall -- --nocapture"`
Expected: `test=ok`

## Chunk 4: Durable State

### Task 4: Mark the feature and queue the next one

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `memory/CURRENT_WORK_ORDER.json`

- [x] **Step 1: Update durable status**

Mark `ivfpq-hot-path-audit` passing and set the next feature from the remaining unlocked queue.

- [x] **Step 2: Validate inventory**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`
