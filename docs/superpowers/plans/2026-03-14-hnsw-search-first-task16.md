# HNSW Search-First Task 16 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce shared generic/bitset HNSW layer-0 L2 distance overhead by batching valid neighbors in groups of four, then re-measure the deterministic bitset diagnosis lane.

**Architecture:** After Task 15 rejected result-container changes, the next `search_first` slice changes mechanism entirely and only targets distance computation. `search_layer_idx_shared(...)` will continue to own traversal, visited checks, filtering, frontier ordering, and result semantics, but on `level=0 + L2 + flat_graph` it will gather valid neighbors into small groups of four and reuse the existing batch-4 L2 helper before feeding distances back into the unchanged accepted-neighbor path.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, integration tests in `tests/`, local `cargo test` plus the deterministic bitset HDF5 diagnosis lane.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-search-first-task16.md`
  - Focused execution plan for shared filtered batch-4 distance.
- Modify: `docs/superpowers/specs/2026-03-14-hnsw-search-first-task16-filtered-batch4-design.md`
  - Reference design for the slice.
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
  - Extend the public contract so the generic shared kernel reports the new shared bitset distance mode.
- Modify: `src/faiss/hnsw.rs`
  - Add focused regressions and batch valid shared-path layer-0 L2 distances in groups of four.
- Modify: `benchmark_results/rs_hnsw_sift128.local.json`
  - Canonical local HNSW artifact after the Task 16 change.
- Modify: `benchmark_results/hnsw_search_cost_diagnosis.json`
  - Canonical local diagnosis artifact after the Task 16 change.
- Modify: `benchmark_results/hnsw_bitset_search_cost_diagnosis.json`
  - Canonical deterministic bitset diagnosis artifact after the Task 16 change.
- Modify: `task-progress.md`
  - Record the local batch-4 screen outcome.
- Modify: `RELEASE_NOTES.md`
  - Record the new shared bitset distance mode and local screen outcome if the code survives.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the local-only result and next recommendation.

## Chunk 1: Lock the Shared Batch-4 Contract

### Task 1: Extend the public generic-kernel report

**Files:**
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing contract expectation**

Extend `generic_kernel_report_declares_shared_idx_traversal()` so it also expects:

```rust
assert_eq!(
    report.shared_bitset_distance_mode,
    "idx_ptr_batch4_when_grouped"
);
```

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: FAIL because the report does not expose the new field yet.

- [ ] **Step 3: Extend `HnswGenericSearchKernelReport`**

Add `shared_bitset_distance_mode` and wire `generic_search_kernel_report()` to the target string.

- [ ] **Step 4: Re-run the contract**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: PASS on the report side.

## Chunk 2: Batch the Shared Layer-0 Distance Path

### Task 2: Add focused regressions and implement narrow batch-4 grouping

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write focused failing regressions**

Add focused library tests that prove:

- the shared generic/bitset layer-0 baseline results still match after batch-4 grouping is introduced
- repeated shared-kernel scratch reuse across calls still behaves correctly
- the shared `level=0 + L2 + flat_graph` path activates the grouped batch-4 distance mode on a deterministic fixture

- [ ] **Step 2: Run the focused regressions to verify red**

Run:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
cargo test test_search_layer_idx_shared_uses_batch4_distance_when_grouping_layer0_neighbors --lib -- --nocapture
```

Expected: the new batch-4 activation regression FAILS before implementation; the existing behavior locks PASS as the baseline.

- [ ] **Step 3: Implement narrow shared-path batch-4 grouping**

In `src/faiss/hnsw.rs`:

- keep `search_layer_idx_shared(...)` as the only traversal core
- only change the `use_layer0_flat_graph` branch for `MetricType::L2`
- accumulate valid, unfiltered neighbor indices in a fixed-size local buffer
- call the existing batch-4 L2 helper for full groups of four
- feed the returned distances through the existing accepted-neighbor logic unchanged
- keep scalar fallback for the tail smaller than four
- keep visited / filter / frontier / result semantics unchanged

- [ ] **Step 4: Re-run the focused regressions**

Run:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
cargo test test_search_layer_idx_shared_uses_batch4_distance_when_grouping_layer0_neighbors --lib -- --nocapture
```

Expected: PASS.

## Chunk 3: Measure the Effect on the Deterministic Bitset Lane

### Task 3: Re-run the canonical local artifact and one same-code rerun

**Files:**
- Modify: `benchmark_results/rs_hnsw_sift128.local.json`
- Modify: `benchmark_results/hnsw_search_cost_diagnosis.json`
- Modify: `benchmark_results/hnsw_bitset_search_cost_diagnosis.json`

- [ ] **Step 1: Capture the new canonical artifact**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.local.json --diagnosis-output benchmark_results/hnsw_search_cost_diagnosis.json --bitset-diagnosis-output benchmark_results/hnsw_bitset_search_cost_diagnosis.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS.

- [ ] **Step 2: Run one same-code `/tmp` rerun**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/rs_hnsw_sift128.task16.rerun2.json --diagnosis-output /tmp/hnsw_search_cost_diagnosis.task16.rerun2.json --bitset-diagnosis-output /tmp/hnsw_bitset_search_cost_diagnosis.task16.rerun2.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS, letting the screen decision compare Task-16 delta against same-code drift.

## Chunk 4: Verify and Record the Local Screen Result

### Task 4: Close the Task 16 screen

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

- the new `shared_bitset_distance_mode`
- the deterministic bitset-lane before/after delta versus the Task-14 canonical baseline
- the same-code rerun drift
- whether this slice is `promote`, `needs_more_local`, or `reject`
