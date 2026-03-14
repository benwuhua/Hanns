# HNSW Search-First Task 12 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared generic/bitset HNSW layer-search kernel use the pointer-specialized L2 distance path when the index metric is L2, then measure the effect on the new deterministic bitset diagnosis lane.

**Architecture:** `search_layer_idx_shared(...)` currently shares traversal logic across generic and bitset search, but it still computes distances through `self.distance(...)` even when the index metric is L2. This slice keeps the shared traversal structure unchanged, adds a narrow public report field for the L2 dispatch mode, and replaces the shared kernel’s L2 distance calls with the existing pointer-specialized helper used elsewhere in HNSW.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, integration tests in `tests/`, local `cargo test` plus the deterministic bitset HDF5 diagnosis lane.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-search-first-task12.md`
  - Focused execution plan for shared-kernel L2 distance specialization.
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
  - Extend the public contract so the report exposes the shared kernel’s L2 distance dispatch mode.
- Modify: `src/faiss/hnsw.rs`
  - Add the public report field and specialize L2 distance lookup inside `search_layer_idx_shared(...)`.
- Modify: `task-progress.md`
  - Record this screen slice and the bitset-lane before/after result.
- Modify: `RELEASE_NOTES.md`
  - Record the new shared-kernel L2 dispatch mode and whether it moved the local bitset lane.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the local-only result and next recommendation.

## Chunk 1: Lock the Shared-Kernel L2 Dispatch Contract

### Task 1: Extend the public report with the shared L2 dispatch mode

**Files:**
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing contract expectation**

Extend `generic_kernel_report_declares_shared_idx_traversal()` so it also expects:

```rust
assert_eq!(report.shared_l2_distance_dispatch, "idx_ptr_kernel");
```

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: FAIL because the report does not expose the new field yet.

- [ ] **Step 3: Extend `HnswGenericSearchKernelReport`**

Add `shared_l2_distance_dispatch` to the public report and wire `generic_search_kernel_report()` to the intended target string.

- [ ] **Step 4: Re-run the contract**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: PASS on the report side; deeper behavior still needs regression coverage.

## Chunk 2: Specialize L2 Distance Lookup Inside the Shared Kernel

### Task 2: Route shared-kernel L2 distance calls through the pointer helper

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Reuse the existing deterministic shared-kernel result tests**

Treat these as the behavior lock for the refactor:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected before implementation: PASS, establishing the current shared-kernel result baseline.

- [ ] **Step 2: Implement L2 specialization in `search_layer_idx_shared(...)`**

Inside `src/faiss/hnsw.rs`:

- detect `self.metric_type == MetricType::L2`
- precompute `query_ptr` and `base_ptr`
- compute entry and neighbor distances via `l2_distance_to_idx_ptr(...)`
- keep the existing generic `self.distance(...)` path for non-L2 metrics
- preserve bitset filtering semantics, frontier/result ordering, and profile accounting

- [ ] **Step 3: Re-run the shared-kernel behavior tests**

Run:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected: PASS.

## Chunk 3: Measure the Task 12 Effect on the Dedicated Bitset Lane

### Task 3: Re-run the deterministic bitset diagnosis lane

**Files:**
- Modify: `benchmark_results/hnsw_bitset_search_cost_diagnosis.json`
- Modify: `benchmark_results/rs_hnsw_sift128.local.json`
- Modify: `benchmark_results/hnsw_search_cost_diagnosis.json`

- [ ] **Step 1: Capture the new canonical bitset diagnosis artifact**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.local.json --diagnosis-output benchmark_results/hnsw_search_cost_diagnosis.json --bitset-diagnosis-output benchmark_results/hnsw_bitset_search_cost_diagnosis.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS.

- [ ] **Step 2: Run one same-code `/tmp` rerun**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/rs_hnsw_sift128.task12.rerun2.json --diagnosis-output /tmp/hnsw_search_cost_diagnosis.task12.rerun2.json --bitset-diagnosis-output /tmp/hnsw_bitset_search_cost_diagnosis.task12.rerun2.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS, letting the screen decision compare the before/after delta against the lane’s own same-code drift.

## Chunk 4: Verify and Record the Local Screen Result

### Task 4: Close the Task 12 screen

**Files:**
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Re-run coverage and workflow checks**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
cargo test --test bench_hnsw_search_first_bitset_diagnosis -- --nocapture
cargo test hnsw --lib -- --nocapture
cargo fmt --all -- --check
python3 scripts/validate_features.py feature-list.json
```

Expected: PASS.

- [ ] **Step 2: Record the screen result**

Update durable docs with:

- the new L2 dispatch mode in the shared kernel
- the bitset-lane before/after delta
- the same-code rerun drift
- whether this slice stays `needs_more_local` or becomes a promotable local signal
