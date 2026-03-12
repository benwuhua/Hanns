# Prod FFI Observability Persistence Gate Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans or superpowers:subagent-driven-development before changing code from this plan.

**Goal:** Close the remote cross-cutting FFI / observability / persistence gate with fresh authority evidence and synced durable state.

**Architecture:** Treat the feature as a verification-first closure. The local test lanes act only as a prefilter. The actual acceptance evidence is the remote x86 replay of the three recorded contract surfaces, using isolated remote target/log directories and serialized execution to avoid shared-lock conflicts.

**Tech Stack:** Rust 2021, cargo tests, remote authority wrappers in `scripts/remote/`, durable long-task state files

---

## Chunk 1: Reconfirm The Existing Contract Surfaces

### Task 1: Run the recorded local lanes

**Files:**
- Inspect: `src/ffi.rs`
- Inspect: `src/serialize.rs`
- Inspect: `tests/bench_json_export.rs`

- [ ] **Step 1: Run the local FFI lane**

Run: `cargo test --lib ffi -- --nocapture`
Expected: PASS

- [ ] **Step 2: Run the local serialize lane**

Run: `cargo test --lib serialize -- --nocapture`
Expected: PASS

- [ ] **Step 3: Run the local JSON export lane**

Run: `cargo test --test bench_json_export -q`
Expected: PASS

## Chunk 2: Refresh Authority Evidence

### Task 2: Replay the same surfaces on remote x86

**Files:**
- Inspect: `scripts/remote/test.sh`
- Inspect: `scripts/remote/common.sh`

- [ ] **Step 1: Refresh the remote workspace**

Run: `bash init.sh`
Expected: PASS

- [ ] **Step 2: Use isolated remote target/log directories**

Set:

- `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-prod-ffi-observability-persistence-gate`
- `KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-prod-ffi-observability-persistence-gate`

- [ ] **Step 3: Replay the remote FFI lane**

Run: `bash scripts/remote/test.sh --command "cargo test --lib ffi -- --nocapture"`
Expected: `test=ok`

- [ ] **Step 4: Replay the remote serialize lane**

Run: `bash scripts/remote/test.sh --command "cargo test --lib serialize -- --nocapture"`
Expected: `test=ok`

- [ ] **Step 5: Replay the remote JSON export lane**

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_json_export -q"`
Expected: `test=ok`

Note: if a parallel or stale run hits `status=conflict`, discard it and rerun serially. Only the serialized isolated runs count as authority evidence.

## Chunk 3: Sync Durable State

### Task 3: Mark the feature passing and advance the queue

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Mark the feature passing**

Update `feature-list.json` so `prod-ffi-observability-persistence-gate` is `passing`.

- [ ] **Step 2: Record the authority logs and next feature**

Update the durable docs so they:

- capture the three remote log paths
- note the initial lock-conflict behavior as non-authoritative scheduler noise
- move the next ready feature to `prod-readme-remote-workflow-docs`

- [ ] **Step 3: Validate durable consistency**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`
