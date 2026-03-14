# HNSW Search-First Task 11 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated local bitset/generic diagnosis lane so future `search_first` slices can measure the non-fast-path HNSW kernel on a deterministic workload instead of relying only on the layer-0 L2 fast-path-heavy HDF5 lane.

**Architecture:** The existing `benchmark_results/hnsw_search_cost_diagnosis.json` lane only measures the unfiltered L2 path and is dominated by the layer-0 ordered-pool fast path. This slice adds a narrow profiled bitset-search diagnosis surface in `src/faiss/hnsw.rs`, extends the HDF5 generator with a deterministic bitset replay mode, and writes a new local-only artifact for filtered/generic search-cost analysis.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, `src/bin/generate_hdf5_hnsw_baseline.rs`, integration tests in `tests/`, local `cargo test` plus local `cargo run --release --features hdf5`.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-search-first-task11.md`
  - Focused execution plan for the dedicated bitset diagnosis lane.
- Create: `tests/bench_hnsw_search_first_bitset_diagnosis.rs`
  - Contract test that requires the new local bitset diagnosis artifact and locks its schema.
- Modify: `src/faiss/hnsw.rs`
  - Add a profiled bitset search-cost diagnosis surface and focused unit coverage.
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`
  - Add deterministic bitset replay options and write the new diagnosis artifact.
- Modify: `task-progress.md`
  - Record this session as the next `search_first` screen slice.
- Modify: `RELEASE_NOTES.md`
  - Record that a dedicated bitset diagnosis lane now exists.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the local-only diagnosis result and what it implies for the next slice.

## Chunk 1: Lock the New Artifact Contract

### Task 1: Add a default-lane contract for the bitset diagnosis artifact

**Files:**
- Create: `tests/bench_hnsw_search_first_bitset_diagnosis.rs`

- [ ] **Step 1: Write the failing artifact contract**

Create `tests/bench_hnsw_search_first_bitset_diagnosis.rs` with a JSON loader for `benchmark_results/hnsw_bitset_search_cost_diagnosis.json` and assertions that:

```rust
assert_eq!(artifact["benchmark"], "HNSW-SEARCH-FIRST-bitset-search-cost-diagnosis");
assert!(artifact["ef_sweep"].is_array());
assert!(artifact["bitset_stride"].is_number());
assert!(artifact["filtered_fraction"].is_number());
assert!(artifact["selected_recall_gate"].is_number());
```

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_search_first_bitset_diagnosis -- --nocapture
```

Expected: FAIL because the new artifact does not exist yet.

## Chunk 2: Add the Profile API

### Task 2: Expose bitset search-cost diagnosis from `HnswIndex`

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing unit test**

Near the existing HNSW tests, add:

```rust
#[test]
fn test_search_cost_diagnosis_with_bitset_reports_visited_nodes() {
    use crate::bitset::BitsetView;

    let index = deterministic_upper_layer_index();
    let query = [11.8, 0.0];
    let mut bitset = BitsetView::new(index.ntotal());
    bitset.set(0, true);
    bitset.set(2, true);

    let report = index.search_cost_diagnosis_with_bitset(&query, 4, 2, &bitset);

    assert!(report.visited_nodes > 0);
    assert!(report.frontier_pops > 0);
    assert!(report.distance_calls > 0);
}
```

- [ ] **Step 2: Run the test to verify red**

Run:

```bash
cargo test test_search_cost_diagnosis_with_bitset_reports_visited_nodes --lib -- --nocapture
```

Expected: FAIL because `search_cost_diagnosis_with_bitset()` does not exist yet.

- [ ] **Step 3: Implement the profiled bitset diagnosis surface**

In `src/faiss/hnsw.rs`, add a new public method:

```rust
pub fn search_cost_diagnosis_with_bitset(
    &self,
    query: &[f32],
    ef: usize,
    k: usize,
    bitset: &BitsetView,
) -> HnswSearchCostDiagnosis
```

Implementation rules:

- reuse the existing `HnswCandidateSearchProfileStats`
- mirror the control flow of `search_single_with_bitset(...)`
- keep it local-only and diagnostic; do not change production search signatures
- preserve the current bitset semantics and result ordering

- [ ] **Step 4: Re-run the unit test**

Run:

```bash
cargo test test_search_cost_diagnosis_with_bitset_reports_visited_nodes --lib -- --nocapture
```

Expected: PASS.

## Chunk 3: Add Deterministic HDF5 Bitset Replay

### Task 3: Generate the dedicated bitset diagnosis artifact

**Files:**
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`
- Create: `benchmark_results/hnsw_bitset_search_cost_diagnosis.json`

- [ ] **Step 1: Extend the CLI with deterministic bitset options**

Add:

- `--bitset-diagnosis-output <path>`
- `--bitset-step <n>`

Behavior:

- if `--bitset-diagnosis-output` is present, build a deterministic bitset that masks every `bitset_step`-th base vector
- require `bitset_step >= 2`

- [ ] **Step 2: Add filtered exact-ground-truth support**

Extend the exact-ground-truth helper so it can skip masked base vectors when building filtered recall targets for the bitset lane.

- [ ] **Step 3: Add repeated bitset query execution**

Add a `run_query_batch_with_bitset_repeated(...)` helper that mirrors the unfiltered repeated runner but calls `index.search_with_bitset(...)`.

- [ ] **Step 4: Write the bitset diagnosis artifact**

Emit `benchmark_results/hnsw_bitset_search_cost_diagnosis.json` with:

- benchmark name
- dataset
- methodology
- `build_random_seed`
- `qps_repeat_count`
- `bitset_stride`
- `filtered_fraction`
- `selected_recall_gate`
- `ef_sweep` rows with qps, qps_runs, recall, visited/frontier/distance averages

- [ ] **Step 5: Run the generator**

Run:

```bash
cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.local.json --diagnosis-output benchmark_results/hnsw_search_cost_diagnosis.json --bitset-diagnosis-output benchmark_results/hnsw_bitset_search_cost_diagnosis.json --bitset-step 3 --ef-sweep 64,96,128,160 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95 --random-seed 42 --repeat 3
```

Expected: PASS and writes the new artifact.

## Chunk 4: Verify and Record the Local Screen Result

### Task 4: Close the Task 11 local screen

**Files:**
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Run the new artifact contract and HNSW coverage**

Run:

```bash
cargo test --test bench_hnsw_search_first_bitset_diagnosis -- --nocapture
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

- [ ] **Step 3: Record the screen result**

Update `task-progress.md`, `RELEASE_NOTES.md`, and `docs/PARITY_AUDIT.md` with:

- the new bitset diagnosis artifact path
- the exact local command used to generate it
- whether the new lane is stable enough to guide future `search_first` work
- what kernel slice should be taken next now that a generic/bitset lane exists
