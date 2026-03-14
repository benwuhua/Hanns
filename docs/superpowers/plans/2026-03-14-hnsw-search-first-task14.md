# HNSW Search-First Task 14 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce shared generic/bitset HNSW result-threshold overhead by caching the current worst accepted result in `SearchScratch`, then re-measure the deterministic bitset diagnosis lane.

**Architecture:** After Task 13, `search_layer_idx_shared(...)` already uses flat layer-0 L2 adjacency and pointer-specialized distance dispatch, but it still reads `generic_results.peek()` in the hot loop to recover the current worst accepted distance. This slice keeps the existing result heap as the source of truth, adds a scratch-owned cached worst-threshold for the shared generic/bitset kernel, and routes threshold checks through that cache while preserving result ordering and filtering semantics.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, integration tests in `tests/`, local `cargo test` plus the deterministic bitset HDF5 diagnosis lane.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-search-first-task14.md`
  - Focused execution plan for shared-kernel result-threshold reuse.
- Modify: `docs/superpowers/specs/2026-03-14-hnsw-search-first-task14-frontier-result-reuse-design.md`
  - Reference design for the slice.
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
  - Extend the public contract so the generic shared kernel reports its result-threshold mode.
- Modify: `src/faiss/hnsw.rs`
  - Add the scratch-owned cached worst-threshold, use it inside `search_layer_idx_shared(...)`, and add focused library regressions.
- Modify: `task-progress.md`
  - Record the local bitset-lane before/after comparison.
- Modify: `RELEASE_NOTES.md`
  - Record the new shared result-threshold mode and its local screen outcome.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the local-only result and next recommendation.

## Chunk 1: Lock the Shared Result-Threshold Contract

### Task 1: Extend the public generic-kernel report

**Files:**
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing contract expectation**

Extend `generic_kernel_report_declares_shared_idx_traversal()` so it also expects:

```rust
assert_eq!(
    report.shared_result_threshold_mode,
    "scratch_cached_worst"
);
```

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: FAIL because the report does not expose the new field yet.

- [ ] **Step 3: Extend `HnswGenericSearchKernelReport`**

Add `shared_result_threshold_mode` and wire `generic_search_kernel_report()` to the target string.

- [ ] **Step 4: Re-run the contract**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: PASS on the report side.

## Chunk 2: Cache the Worst Accepted Result in Scratch

### Task 2: Add focused regressions and implement the cache

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write focused failing regressions**

Add focused library tests that prove:

- repeated shared-kernel calls on the same `SearchScratch` do not leak a stale worst-threshold across calls
- shared generic and shared bitset layer-search still match the existing layer-0 baseline results after the cache is introduced

- [ ] **Step 2: Run the focused regressions to verify red**

Run:

```bash
cargo test test_search_layer_idx_shared_cached_worst_threshold_resets_across_calls --lib -- --nocapture
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected: the new cached-threshold regression FAILS before implementation; the existing behavior locks PASS as the baseline.

- [ ] **Step 3: Implement the cached worst-threshold**

In `src/faiss/hnsw.rs`:

- extend `SearchScratch` with a cached shared-kernel worst-result distance
- reset that cache in the generic-heap preparation path
- initialize it from entry insertion when the first unfiltered result is accepted
- refresh it only when `generic_results` actually changes
- route early-termination and accepted-neighbor threshold checks through the cached value instead of repeated `generic_results.peek()` unpacking
- keep `generic_results` as the source of truth for output ordering

- [ ] **Step 4: Re-run the focused regressions**

Run:

```bash
cargo test test_search_layer_idx_shared_cached_worst_threshold_resets_across_calls --lib -- --nocapture
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected: PASS.

## Chunk 3: Measure the Effect on the Bitset Diagnosis Lane

### Task 3: Re-run the deterministic bitset lane and compare to Task 13

**Files:**
- Modify: `benchmark_results/hnsw_bitset_search_cost_diagnosis.json`
- Modify: `benchmark_results/rs_hnsw_sift128.local.json`
- Modify: `benchmark_results/hnsw_search_cost_diagnosis.json`

- [ ] **Step 1: Capture the new canonical artifact**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.local.json --diagnosis-output benchmark_results/hnsw_search_cost_diagnosis.json --bitset-diagnosis-output benchmark_results/hnsw_bitset_search_cost_diagnosis.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS.

- [ ] **Step 2: Run one same-code `/tmp` rerun**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/rs_hnsw_sift128.task14.rerun2.json --diagnosis-output /tmp/hnsw_search_cost_diagnosis.task14.rerun2.json --bitset-diagnosis-output /tmp/hnsw_bitset_search_cost_diagnosis.task14.rerun2.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS, letting the screen decision compare Task-14 delta against same-code drift.

## Chunk 4: Verify and Record the Local Screen Result

### Task 4: Close the Task 14 screen

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

- the new `shared_result_threshold_mode`
- the bitset-lane before/after delta versus the Task-13 baseline
- the same-code rerun drift
- whether this slice stays `needs_more_local` or becomes another promotable local signal
