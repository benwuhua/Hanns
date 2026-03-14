# HNSW Search-First Task 9 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify the generic HNSW layer-search traversal so unfiltered KNN and bitset-filtered KNN share one `idx`-based scratch-backed core without changing current search semantics.

**Architecture:** The current `src/faiss/hnsw.rs` still carries two near-duplicate traversal loops: `search_layer_idx_with_optional_profile()` and `search_layer_idx_with_bitset_scratch()`. This plan introduces one shared layer-search kernel plus a narrow policy enum for result eligibility, then routes both existing call sites through it while keeping the L2 ordered-pool fast path unchanged.

**Tech Stack:** Rust 2021, `src/faiss/hnsw.rs`, integration tests in `tests/`, local cargo test verification only.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-search-first-task9.md`
  - Focused execution plan for the first `search_first` code slice.
- Create: `tests/bench_hnsw_generic_search_kernel.rs`
  - Contract tests that prove the generic and bitset layer-search paths now report the same shared traversal mode and preserve current result behavior.
- Modify: `src/faiss/hnsw.rs`
  - Add the shared layer-search kernel, a minimal report surface, and route current generic/bitset paths through it.
- Modify: `task-progress.md`
  - Record this session as the first post-Stage-1 `search_first` execution slice.
- Modify: `RELEASE_NOTES.md`
  - Record that the generic/bitset traversal loops now share one scratch-backed kernel.
- Modify: `docs/PARITY_AUDIT.md`
  - Record that post-Stage-1 implementation has started on the `search_first` branch.

## Chunk 1: Lock the Shared-Kernel Contract

### Task 1: Add an integration contract for the shared generic kernel

**Files:**
- Create: `tests/bench_hnsw_generic_search_kernel.rs`
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Write the failing contract tests**

Create `tests/bench_hnsw_generic_search_kernel.rs` with:

```rust
#[test]
fn generic_kernel_report_declares_shared_idx_traversal() {
    let index = build_fixture();
    let report = index.generic_search_kernel_report();
    assert_eq!(report.unfiltered_layer_search_core, "shared_idx_binary_heap");
    assert_eq!(report.bitset_layer_search_core, "shared_idx_binary_heap");
    assert_eq!(report.visited_reuse_scope, "visited_epoch_only");
}
```

```rust
#[test]
fn generic_kernel_report_preserves_bitset_screen_fixture_results() {
    let index = build_fixture();
    let bitset = make_fixture_bitset(index.ntotal());
    let query = [0.1, 0.0];
    let req = SearchRequest::new(3, 8);

    let approx = index.search_with_bitset(&query, &req, &bitset).unwrap();
    assert_eq!(approx.ids, vec![1, 3, 5]);
}
```

- [ ] **Step 2: Run the contract to confirm red**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: FAIL because `generic_search_kernel_report()` does not exist yet.

- [ ] **Step 3: Add a minimal public report struct**

In `src/faiss/hnsw.rs`, add:

```rust
#[derive(Clone, Debug, Serialize)]
pub struct HnswGenericSearchKernelReport {
    pub unfiltered_layer_search_core: String,
    pub bitset_layer_search_core: String,
    pub frontier_container: String,
    pub result_container: String,
    pub visited_reuse_scope: String,
}
```

- [ ] **Step 4: Add `generic_search_kernel_report()` with the current target strings**

Return one stable report that future refactors can keep honest. Use strings that describe the actual intended shared kernel shape rather than historical implementation fragments.

- [ ] **Step 5: Re-run the contract**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: the result-parity test passes, but the shared-kernel mode test still fails until the internal refactor is wired.

## Chunk 2: Extract the Shared Traversal Core

### Task 2: Introduce one shared `idx`-based layer-search kernel

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add focused library tests for the new shared helper**

Near the existing HNSW tests, add:

```rust
#[test]
fn test_shared_layer_search_matches_generic_layer0_results() {
    let index = deterministic_parallel_build_entry_descent_fixture();
    let query = vec![102.5, 0.0];
    let mut scratch = SearchScratch::new();
    let expected = index.search_layer_idx_with_scratch(&query, 0, 0, 4, &mut scratch);

    let mut shared_scratch = SearchScratch::new();
    let actual = index.search_layer_idx_shared_for_test(&query, 0, 0, 4, None, &mut shared_scratch);
    assert_eq!(actual, expected);
}
```

```rust
#[test]
fn test_shared_layer_search_matches_bitset_layer0_results() {
    let index = deterministic_parallel_build_entry_descent_fixture();
    let query = vec![102.5, 0.0];
    let mut bitset = BitsetView::new(index.ntotal());
    bitset.set(2, true);
    let mut scratch = SearchScratch::new();
    let expected =
        index.search_layer_idx_with_bitset_scratch(&query, 0, 0, 4, &bitset, &mut scratch);

    let mut shared_scratch = SearchScratch::new();
    let actual =
        index.search_layer_idx_shared_for_test(&query, 0, 0, 4, Some(&bitset), &mut shared_scratch);
    assert_eq!(actual, expected);
}
```

- [ ] **Step 2: Run the targeted tests to confirm red**

Run:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected: FAIL because the shared helper does not exist yet.

- [ ] **Step 3: Implement the shared helper**

Add one internal helper in `src/faiss/hnsw.rs` with this boundary:

```rust
fn search_layer_idx_shared(
    &self,
    query: &[f32],
    entry_idx: usize,
    level: usize,
    ef: usize,
    bitset: Option<&BitsetView>,
    scratch: &mut SearchScratch,
    profile: Option<&mut HnswCandidateSearchProfileStats>,
) -> Vec<(usize, f32)>
```

Rules:

- keep `idx`-based traversal and `SearchScratch`
- keep `BinaryHeap` frontier/results for this generic kernel slice
- if `bitset` is present, allow a filtered entry point to seed the frontier but never admit filtered nodes to `results`
- keep profile accounting only where it already exists for the unfiltered path; do not invent a new stats model in this slice
- leave the L2 ordered-pool fast path untouched

- [ ] **Step 4: Rewire current callers**

Route these methods through the shared helper:

- `search_layer_idx_with_optional_profile()`
- `search_layer_idx_with_scratch()`
- `search_layer_idx_with_bitset_scratch()`

Do not change their external signatures in this slice.

- [ ] **Step 5: Add a small test-only wrapper if needed**

If integration tests cannot reach the shared helper directly, expose only a narrow `#[cfg(test)]` wrapper such as `search_layer_idx_shared_for_test()`.

- [ ] **Step 6: Re-run the targeted tests**

Run:

```bash
cargo test test_shared_layer_search_matches_generic_layer0_results --lib -- --nocapture
cargo test test_shared_layer_search_matches_bitset_layer0_results --lib -- --nocapture
```

Expected: PASS.

## Chunk 3: Close the Contract and Verify Regressions

### Task 3: Close the shared-kernel contract and local regression set

**Files:**
- Modify: `tests/bench_hnsw_generic_search_kernel.rs`
- Modify: `src/faiss/hnsw.rs`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Make the report reflect the actual refactor**

Update `generic_search_kernel_report()` so both unfiltered and bitset paths truthfully report the same shared traversal core.

- [ ] **Step 2: Re-run the integration contract**

Run:

```bash
cargo test --test bench_hnsw_generic_search_kernel -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Re-run HNSW library coverage**

Run:

```bash
cargo test hnsw --lib -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Re-run format check**

Run:

```bash
cargo fmt --all -- --check
```

Expected: PASS.

- [ ] **Step 5: Update durable notes**

Record:

- this is the first code-bearing `search_first` slice after the Stage-1 decision gate
- the refactor is still local-only and not an authority claim
- the next likely follow-up is Task 10 (`re-measure per-visited-node cost`)

