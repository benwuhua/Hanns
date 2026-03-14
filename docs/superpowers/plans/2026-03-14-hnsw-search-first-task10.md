# HNSW Search-First Task 10 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse scratch-backed frontier/result heaps across generic and upper-layer L2 HNSW layer search so the shared search kernel stops allocating fresh `BinaryHeap`s on every call.

**Architecture:** `src/faiss/hnsw.rs` already shares one generic `idx + SearchScratch + BinaryHeap` traversal core for unfiltered and bitset-filtered layer search, but it still allocates fresh frontier/result heaps inside both the generic shared kernel and the upper-layer L2 heap kernel. This slice moves those heaps into `SearchScratch`, adds a narrow public report surface that records the broader reuse scope, and keeps the existing layer-search semantics and the layer-0 ordered-pool fast path unchanged.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, integration tests in `tests/`, local `cargo test` plus the seeded local HDF5 diagnosis lane.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-search-first-task10.md`
  - Focused execution plan for scratch-backed generic heap reuse.
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
  - Extend the outer contract so the public report reflects the broader scratch reuse scope.
- Modify: `src/faiss/hnsw.rs`
  - Add reusable scratch-owned generic frontier/result heaps and route both generic shared search and upper-layer L2 heap search through them.
- Modify: `task-progress.md`
  - Record this session as the second post-Stage-1 `search_first` implementation slice.
- Modify: `RELEASE_NOTES.md`
  - Record that the generic/L2 heap kernels now reuse scratch-owned heaps.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the local screen status and what changed in the pure-Rust search kernel.

## Chunk 1: Lock the Reuse Contract

### Task 1: Extend the shared-kernel contract to describe heap reuse

**Files:**
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing contract expectation**

In `tests/bench_hnsw_generic_search_kernel.rs`, extend `generic_kernel_report_declares_shared_idx_traversal()` so it also expects:

```rust
assert_eq!(report.frontier_reuse_scope, "scratch_binary_heap");
assert_eq!(report.result_reuse_scope, "scratch_binary_heap");
assert_eq!(report.visited_reuse_scope, "visited_epoch_and_generic_heaps");
```

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: FAIL because the public report does not expose the broader reuse scope yet.

- [ ] **Step 3: Extend the public report surface**

In `src/faiss/hnsw.rs`, add `frontier_reuse_scope` and `result_reuse_scope` to `HnswGenericSearchKernelReport`, then update `generic_search_kernel_report()` to report scratch-backed reuse.

- [ ] **Step 4: Re-run the contract**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: the report assertions pass after implementation, while deeper reuse behavior still needs library coverage.

## Chunk 2: Move Generic Heaps into `SearchScratch`

### Task 2: Reuse scratch-owned heaps in generic and upper-layer L2 heap kernels

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add focused failing library tests**

Near the existing HNSW tests, add:

```rust
#[test]
fn test_search_layer_idx_shared_reuses_generic_heap_capacity_across_calls() {
    let index = deterministic_upper_layer_index();
    let query = [11.8, 0.0];
    let mut scratch = SearchScratch::new();

    let first = index.search_layer_idx_shared_for_test(&query, 0, 0, 4, None, &mut scratch);
    let first_frontier_cap = scratch.generic_frontier.capacity();
    let first_result_cap = scratch.generic_results.capacity();

    let second = index.search_layer_idx_shared_for_test(&query, 0, 0, 2, None, &mut scratch);

    assert_eq!(first, second[..first.len().min(second.len())].iter().cloned().take(first.len()).collect::<Vec<_>>());
    assert!(first_frontier_cap >= 8);
    assert!(first_result_cap >= 4);
    assert_eq!(scratch.generic_frontier.capacity(), first_frontier_cap);
    assert_eq!(scratch.generic_results.capacity(), first_result_cap);
}
```

```rust
#[test]
fn test_search_layer_idx_l2_heap_reuses_generic_heap_capacity_across_calls() {
    let index = deterministic_upper_layer_index();
    let query = [11.8, 0.0];
    let mut scratch = SearchScratch::new();

    let _ = index.search_layer_idx_l2_heap_with_scratch(&query, 0, 1, 4, &mut scratch, None);
    let first_frontier_cap = scratch.generic_frontier.capacity();
    let first_result_cap = scratch.generic_results.capacity();

    let _ = index.search_layer_idx_l2_heap_with_scratch(&query, 0, 1, 2, &mut scratch, None);

    assert!(first_frontier_cap >= 8);
    assert!(first_result_cap >= 4);
    assert_eq!(scratch.generic_frontier.capacity(), first_frontier_cap);
    assert_eq!(scratch.generic_results.capacity(), first_result_cap);
}
```

- [ ] **Step 2: Run the targeted tests to verify red**

Run:

```bash
cargo test test_search_layer_idx_shared_reuses_generic_heap_capacity_across_calls --lib -- --nocapture
cargo test test_search_layer_idx_l2_heap_reuses_generic_heap_capacity_across_calls --lib -- --nocapture
```

Expected: FAIL because `SearchScratch` does not own reusable generic heaps yet.

- [ ] **Step 3: Extend `SearchScratch`**

Add scratch-owned generic frontier/result heaps plus a small `prepare_generic_heaps(...)` helper that:

- clears prior state
- preserves capacity across calls
- ensures frontier capacity tracks roughly `ef * 2`
- ensures result capacity tracks `ef`

- [ ] **Step 4: Rewire the generic shared kernel**

Update `search_layer_idx_shared(...)` to use `scratch.generic_frontier` and `scratch.generic_results` instead of allocating fresh `BinaryHeap`s. Preserve:

- filtered-entry semantics
- filtered-neighbor exclusion
- current profile accounting
- sorted nearest-to-farthest output

- [ ] **Step 5: Rewire upper-layer L2 heap search**

Update `search_layer_idx_l2_heap_with_optional_profile(...)` to use the same scratch-owned generic heaps so upper-layer L2 search also stops allocating fresh heaps each call.

- [ ] **Step 6: Re-run the targeted tests**

Run:

```bash
cargo test test_search_layer_idx_shared_reuses_generic_heap_capacity_across_calls --lib -- --nocapture
cargo test test_search_layer_idx_l2_heap_reuses_generic_heap_capacity_across_calls --lib -- --nocapture
```

Expected: PASS.

## Chunk 3: Verify and Refresh Local Screen Artifacts

### Task 3: Close the local verification loop for Task 10

**Files:**
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `benchmark_results/hnsw_search_cost_diagnosis.json`
- Modify: `benchmark_results/rs_hnsw_sift128.local.json`

- [ ] **Step 1: Re-run the outer contract and HNSW coverage**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
cargo test hnsw --lib -- --nocapture
```

Expected: PASS.

- [ ] **Step 2: Re-run formatting and workflow validation**

Run:

```bash
cargo fmt --all -- --check
python3 scripts/validate_features.py feature-list.json
```

Expected: PASS.

- [ ] **Step 3: Re-run the seeded repeat-3 diagnosis lane**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.local.json --diagnosis-output benchmark_results/hnsw_search_cost_diagnosis.json --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS with updated local-only diagnosis artifact.

- [ ] **Step 4: Record the local screen result**

Update `task-progress.md`, `RELEASE_NOTES.md`, and `docs/PARITY_AUDIT.md` with:

- what changed in the generic/L2 heap kernel
- the local verification commands
- whether the seeded repeat-3 diagnosis lane shows enough signal to keep `screen_result=needs_more_local` or to promote the next slice
