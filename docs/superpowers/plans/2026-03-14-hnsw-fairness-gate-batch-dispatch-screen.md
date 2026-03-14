# HNSW Fairness Gate Batch Dispatch Screen Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in batch-first query dispatch mode to the HDF5 HNSW baseline harness and decide locally whether it is strong enough to promote into the next authority fairness rerun.

**Architecture:** Keep the HNSW index implementation unchanged and add dispatch control only in `generate_hdf5_hnsw_baseline`. The binary will support serial and `rayon` batch-parallel query execution, write accurate dispatch metadata, and then run a local serial-vs-parallel screen on the same HDF5 lane.

**Tech Stack:** Rust 2021, `rayon`, HDF5 benchmark binary in `src/bin/generate_hdf5_hnsw_baseline.rs`, local screen evidence recorded in `task-progress.md`.

---

### Task 1: Lock The Batch-Dispatch Contract

**Files:**
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`

- [ ] **Step 1: Write the failing test**

Add a unit test that:

- builds a tiny deterministic HNSW index
- runs the current serial query helper
- runs a new parallel batch helper with `query_batch_size=2`
- asserts identical ordered result IDs
- asserts the dispatch metadata for the parallel helper is non-serial and reports batch size `2`

- [ ] **Step 2: Run test to verify it fails**

Run:
`cargo test --features hdf5 --bin generate_hdf5_hnsw_baseline parallel_query_dispatch_matches_serial_results_and_reports_batch_metadata -- --nocapture`

Expected:
- `FAIL` because the dispatch abstraction and metadata support do not exist yet

### Task 2: Implement The Minimal Batch-First Dispatch Path

**Files:**
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`

- [ ] **Step 1: Add dispatch configuration**

Add:

- a dispatch mode enum
- a small config/metadata helper
- CLI parsing for `--query-dispatch-mode` and `--query-batch-size`

Keep the default mode `serial`.

- [ ] **Step 2: Implement batch-first parallel query execution**

Update the query-running helpers so they can execute either:

- serial per-query search
- `rayon` chunk-parallel batched search

Preserve result ordering.

- [ ] **Step 3: Wire artifact metadata**

Update the emitted `query_dispatch_model` and `query_batch_size` fields so they come from the chosen dispatch config.

- [ ] **Step 4: Run the focused test to green**

Run:
`cargo test --features hdf5 --bin generate_hdf5_hnsw_baseline parallel_query_dispatch_matches_serial_results_and_reports_batch_metadata -- --nocapture`

Expected:
- `PASS`

### Task 3: Re-Run Focused Harness Verification

**Files:**
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`

- [ ] **Step 1: Run existing harness tests**

Run:
`cargo test --features hdf5 --bin generate_hdf5_hnsw_baseline -- --nocapture`

Expected:
- all binary unit tests pass

- [ ] **Step 2: Run fairness metadata smoke**

Run:
`python3 -m unittest tests/test_hnsw_fairness_gate.py`

Expected:
- current durable fairness tests still pass because the default mode remains serial

### Task 4: Execute The Local Screen

**Files:**
- Modify: `task-progress.md`

- [ ] **Step 1: Run the serial baseline**

Run:
`cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/hnsw_fairness_dispatch_serial.json --base-limit 100000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --hnsw-adaptive-k 0 --recall-gate 0.95 --random-seed 42`

- [ ] **Step 2: Run the parallel batch baseline**

Run:
`cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/hnsw_fairness_dispatch_parallel.json --base-limit 100000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --hnsw-adaptive-k 0 --query-dispatch-mode parallel --query-batch-size 32 --recall-gate 0.95 --random-seed 42`

- [ ] **Step 3: Compare the two local outputs**

Confirm:

- parallel artifact reports non-serial dispatch metadata
- recall matches serial
- qps is not an obvious regression

- [ ] **Step 4: Record the screen result**

Update `task-progress.md` with:

- commands run
- serial vs parallel qps/recall
- `screen_result=promote|needs_more_local|reject`
- recommended next step

### Task 5: Final Verification

**Files:**
- Modify: `task-progress.md`

- [ ] **Step 1: Run validator**

Run:
`python3 scripts/validate_features.py feature-list.json`

Expected:
- `VALID - 66 features (66 passing, 0 failing); workflow/doc checks passed`

- [ ] **Step 2: Check worktree**

Run:
`git status --short`

Expected:
- only files from this screen slice are modified before commit
