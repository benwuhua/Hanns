# HNSW Search-First Task 13 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shared generic/bitset HNSW search kernel use `layer0_flat_graph` neighbor IDs for L2 layer-0 traversal when the flat graph is available, then re-measure the deterministic bitset diagnosis lane.

**Architecture:** After Task 12, `search_layer_idx_shared(...)` already uses pointer-specialized L2 distance dispatch, but on layer 0 it still walks canonical `layer_neighbors[level].ids` and pays `id -> idx` conversion on every neighbor. This slice keeps the shared heap-based traversal intact, adds a narrow report field for the shared layer-0 neighbor layout, and switches the L2 level-0 branch to `layer0_flat_graph.neighbors_for(...)` when possible.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, integration tests in `tests/`, local `cargo test` plus the deterministic bitset HDF5 diagnosis lane.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-search-first-task13.md`
  - Focused execution plan for shared-kernel layer-0 flat-graph access.
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
  - Extend the public contract so the generic shared kernel reports its layer-0 neighbor layout.
- Modify: `src/faiss/hnsw.rs`
  - Add the public report field and route L2 layer-0 shared search through `layer0_flat_graph` when enabled.
- Modify: `task-progress.md`
  - Record the local bitset-lane before/after comparison.
- Modify: `RELEASE_NOTES.md`
  - Record the new shared layer-0 neighbor layout mode and its local screen outcome.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the local-only result and next recommendation.

## Chunk 1: Lock the Shared Layer-0 Neighbor Layout Contract

### Task 1: Extend the public generic-kernel report

**Files:**
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing contract expectation**

Extend `generic_kernel_report_declares_shared_idx_traversal()` so it also expects:

```rust
assert_eq!(
    report.shared_layer0_neighbor_layout,
    "flat_u32_adjacency_when_enabled"
);
```

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: FAIL because the report does not expose the new field yet.

- [ ] **Step 3: Extend `HnswGenericSearchKernelReport`**

Add `shared_layer0_neighbor_layout` and wire `generic_search_kernel_report()` to the target string.

- [ ] **Step 4: Re-run the contract**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: PASS on the report side.

## Chunk 2: Switch Shared L2 Layer-0 Traversal to Flat-Graph Neighbors

### Task 2: Use `layer0_flat_graph` neighbor IDs when eligible

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Reuse the existing shared-kernel behavior locks**

Run:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected before implementation: PASS, establishing the current result baseline.

- [ ] **Step 2: Implement the flat-graph branch**

Inside `search_layer_idx_shared(...)`:

- detect `level == 0`
- require `self.metric_type == MetricType::L2`
- require `self.layer0_flat_graph.is_enabled_for(num_nodes)`
- iterate neighbors via `layer0_flat_graph.neighbors_for(cand_idx)` and use the `u32` indices directly
- keep the existing canonical `layer_neighbors[level].ids` path for non-L2 or non-flat-graph cases
- preserve bitset semantics, frontier/result ordering, and profile accounting

- [ ] **Step 3: Re-run the behavior locks**

Run:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected: PASS.

## Chunk 3: Measure the Effect on the Bitset Diagnosis Lane

### Task 3: Re-run the deterministic bitset lane and compare to Task 12

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
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/rs_hnsw_sift128.task13.rerun2.json --diagnosis-output /tmp/hnsw_search_cost_diagnosis.task13.rerun2.json --bitset-diagnosis-output /tmp/hnsw_bitset_search_cost_diagnosis.task13.rerun2.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS, letting the screen decision compare Task-13 delta against same-code drift.

## Chunk 4: Verify and Record the Local Screen Result

### Task 4: Close the Task 13 screen

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

- the new `shared_layer0_neighbor_layout`
- the bitset-lane before/after delta
- the same-code rerun drift
- whether this slice stays `needs_more_local` or becomes a promotable local signal
