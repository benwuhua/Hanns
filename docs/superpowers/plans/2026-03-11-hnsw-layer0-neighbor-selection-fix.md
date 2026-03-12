# HNSW Layer-0 Neighbor Selection Fix Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the smallest possible FAISS-style layer-0 backfill behavior to Rust HNSW construction and prove it improves graph quality without breaking the new regression floor.

**Architecture:** Keep the existing hnswlib-style diversification logic as the default path, but add a layer-aware variant that can refill pruned outsiders only for layer 0 up to `m_max0`. Reuse existing insertion and shrink paths, limit behavior change to forward layer-0 neighbor selection, and validate via unit tests plus authority benchmark lanes.

**Tech Stack:** Rust 2021, cargo test, existing HNSW test suite, remote authority wrappers in `scripts/remote/test.sh`

---

## Chunk 1: Layer-0 TDD

### Task 1: Add failing unit coverage for layer-0 backfill

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing test**

Add a unit test next to the existing audit fixture that asserts layer-0 selection refills pruned outsiders to `m_max0`, while non-layer-0 selection keeps the no-backfill behavior.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_layer0_neighbor_selection_matches_faiss_backfill --lib -- --nocapture`
Expected: FAIL because current `select_neighbors_heuristic_idx` does not backfill.

- [ ] **Step 3: Write minimal implementation**

Add a layer-aware heuristic helper and switch insertion on `level == 0` to use the backfill variant only there.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test test_layer0_neighbor_selection_matches_faiss_backfill --lib -- --nocapture`
Expected: PASS

## Chunk 2: Regression Safety

### Task 2: Re-run HNSW regression floor locally

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Verify: `tests/test_hnsw_advanced_paths.rs`
- Verify: `tests/debug_graph_connectivity.rs`

- [ ] **Step 1: Run focused regressions**

Run:
- `cargo test --test test_hnsw_advanced_paths test_hnsw_build_quality_signals_survive_save_load -- --nocapture`
- `cargo test --test debug_graph_connectivity -- --nocapture`

Expected: PASS

## Chunk 3: Authority Verification

### Task 3: Re-run feature verification commands on authority

**Files:**
- Verify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Sync workspace**

Run: `bash init.sh`
Expected: remote authority ready

- [ ] **Step 2: Run authority lib HNSW lane**

Run: `bash scripts/remote/test.sh --command "cargo test --lib hnsw -- --nocapture"`
Expected: `test=ok`

- [ ] **Step 3: Run authority benchmark comparison lane**

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"`
Expected: `test=ok`

- [ ] **Step 4: Run authority recall lane**

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_recall -q"`
Expected: `test=ok`

## Chunk 4: Durable State

### Task 4: Mark feature complete if all verification passes

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `memory/CURRENT_WORK_ORDER.json`

- [ ] **Step 1: Update durable status**

Mark `hnsw-layer0-neighbor-selection-fix` passing and set the next feature.

- [ ] **Step 2: Validate inventory**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`
