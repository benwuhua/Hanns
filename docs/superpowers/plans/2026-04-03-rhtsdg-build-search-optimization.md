# RHTSDG Build and Search Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce RHTSDG build and search cost in `knowhere-rs` while preserving the current high-recall authority lane.

**Architecture:** Keep optimization work inside the existing `src/faiss/rhtsdg/*` slice and run it as a sequence of narrow `screen` hypotheses. Treat build and search as one combined optimization program: first add enough instrumentation to isolate phase costs, then optimize build hot paths in `xndescent.rs` / `neighbor.rs` / `tsdg.rs`, then optimize search hot paths in `search.rs`. Only promote a bundle to authority if it clears the current trusted local baseline on the representative lane.

**Tech Stack:** Rust 2021, `rayon`, existing SIMD helpers in `crate::simd`, local cargo test/release screens, remote `bash init.sh` plus `scripts/remote/test.sh` for authority verification

---

## Scope Notes

- This plan covers both build and search optimization for the Rust-core RHTSDG implementation.
- This plan intentionally excludes `src/ffi.rs`, `src/python/mod.rs`, and `src/jni/mod.rs`.
- This plan intentionally excludes shortcut-graph work until the current build/search implementation is materially faster on the recorded lane.
- This plan must respect the repository workflow: `screen -> authority -> durable closure`.
- This plan must use session 241 as the best trusted comparison point until a better lane is proven:
  - local: `rhtsdg build_s=10.477, search_s=0.018, qps=10942.39, recall@10=0.9965`
  - authority: `rhtsdg build_s=17.902, search_s=0.039, qps=5066.65, recall@10=0.9965`
- This plan should not reopen known local no-go hypotheses unless new evidence appears:
  - cross-call heap reuse
  - singleton pointer-kernel follow-up
  - flat-layer adjacency storage
  - upper-layer greedy descent

## Planned File Map

**Modify**
- `src/faiss/rhtsdg/search.rs`: search instrumentation, layer-0 traversal hot path, result/frontier maintenance
- `src/faiss/rhtsdg/xndescent.rs`: sampled initialization, join execution, distance evaluation, per-iteration hot loops
- `src/faiss/rhtsdg/neighbor.rs`: neighborhood storage and candidate extraction costs
- `src/faiss/rhtsdg/tsdg.rs`: diversification and reverse-edge filtering costs
- `src/faiss/rhtsdg/config.rs`: optional internal tuning knobs only if needed for a proven optimization
- `tests/test_rhtsdg_screen.rs`: search-side audit fixtures and screen locks
- `tests/test_rhtsdg_xndescent.rs`: neighborhood/build audit fixtures and deterministic build-side locks
- `tests/test_rhtsdg_tsdg.rs`: diversification audit locks when TSDG internals change
- `benches/rhtsdg_bench.rs`: bench lane updates after a surviving optimization bundle exists
- `examples/rhtsdg_vs_hnsw.rs`: representative local/authority comparison lane and optional breakdown output
- `task-progress.md`: log every `screen` result, including reverted no-go experiments and promoted bundles

**Do Not Modify Unless Promotion Is Earned**
- `feature-list.json`
- `RELEASE_NOTES.md`

**Do Not Create**
- any new top-level `src/graph/*`
- any new FFI/JNI/Python surface

## Chunk 1: Measurement and Acceptance Harness

### Task 1: Add test-only build/search phase tracing before changing performance-critical logic

**Files:**
- Modify: `src/faiss/rhtsdg/search.rs`
- Modify: `src/faiss/rhtsdg/xndescent.rs`
- Modify: `src/faiss/rhtsdg/tsdg.rs`
- Modify: `tests/test_rhtsdg_screen.rs`
- Modify: `tests/test_rhtsdg_xndescent.rs`

- [ ] **Step 1: Write the failing trace tests**

```rust
#[test]
fn build_trace_reports_nonzero_xndescent_and_tsdg_phases() {
    let trace = rhtsdg_build_trace_fixture();
    assert!(trace.xndescent_iters > 0);
    assert!(trace.stage1_pairs_checked > 0);
}

#[test]
fn search_trace_reports_layer0_and_upper_layer_work_separately() {
    let trace = layered_search_trace_fixture();
    assert!(trace.layer0_frontier_pops > 0);
    assert!(trace.upper_layer_visits > 0);
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `cargo test --test test_rhtsdg_xndescent build_trace_reports_nonzero_xndescent_and_tsdg_phases -- --nocapture`
Expected: FAIL with missing trace helpers or missing fields

Run: `cargo test --test test_rhtsdg_screen search_trace_reports_layer0_and_upper_layer_work_separately -- --nocapture`
Expected: FAIL with missing trace helpers or missing fields

- [ ] **Step 3: Implement test-only trace structs without changing external API**

```rust
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RhtsdgBuildTrace {
    pub xndescent_iters: usize,
    pub stage1_pairs_checked: usize,
    pub stage2_reverse_candidates: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RhtsdgSearchTrace {
    pub upper_layer_visits: usize,
    pub layer0_frontier_pops: usize,
    pub batch4_calls: usize,
}
```

- [ ] **Step 4: Re-run the focused tests and verify they pass**

Run: `cargo test --test test_rhtsdg_xndescent build_trace_reports_nonzero_xndescent_and_tsdg_phases -- --nocapture`
Expected: PASS

Run: `cargo test --test test_rhtsdg_screen search_trace_reports_layer0_and_upper_layer_work_separately -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit the trace scaffolding**

```bash
git add src/faiss/rhtsdg/search.rs src/faiss/rhtsdg/xndescent.rs src/faiss/rhtsdg/tsdg.rs tests/test_rhtsdg_screen.rs tests/test_rhtsdg_xndescent.rs
git commit -m "test(rhtsdg): add build and search trace fixtures"
```

### Task 2: Lock the stop/go acceptance lane before optimization work begins

**Files:**
- Modify: `examples/rhtsdg_vs_hnsw.rs`
- Modify: `task-progress.md`

- [ ] **Step 1: Add a failing local acceptance note to the plan log**

```md
- current trusted local baseline remains session 241 until a new lane beats both:
  - search_s <= 0.018 or qps >= 10942.39
  - recall@10 >= 0.9965
```

- [ ] **Step 2: Add an opt-in example breakdown path**

```rust
if env::var_os("KNOWHERE_RS_RHTSDG_TRACE").is_some() {
    eprintln!("rhtsdg_trace={:?}", trace_summary);
}
```

- [ ] **Step 3: Run the example once to verify the baseline lane still works**

Run: `cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128`
Expected: PASS with printed `hnsw` and `rhtsdg` summary lines

- [ ] **Step 4: Record the baseline lane in `task-progress.md`**

Run: none
Expected: `task-progress.md` contains the current trusted local/authority comparison point for future stop/go decisions

- [ ] **Step 5: Commit the baseline lane lock**

```bash
git add examples/rhtsdg_vs_hnsw.rs task-progress.md
git commit -m "docs(rhtsdg): lock optimization acceptance lane"
```

## Chunk 2: Build-Side Optimization Screens

### Task 3: Remove layer-local vector copying from hierarchy build and distance access

**Files:**
- Modify: `src/faiss/rhtsdg/search.rs`
- Modify: `src/faiss/rhtsdg/xndescent.rs`
- Modify: `src/faiss/rhtsdg/tsdg.rs`
- Modify: `tests/test_rhtsdg_xndescent.rs`
- Modify: `tests/test_rhtsdg_tsdg.rs`

- [ ] **Step 1: Write the failing build-copy audit tests**

```rust
#[test]
fn hierarchy_build_trace_reports_zero_layer_vector_copy_bytes() {
    let trace = rhtsdg_build_trace_fixture();
    assert_eq!(trace.layer_vector_copy_bytes, 0);
}

#[test]
fn subset_distance_view_matches_copied_vector_distance_order() {
    let fixture = subset_distance_fixture();
    assert_eq!(fixture.borrowed_order(), fixture.copied_order());
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `cargo test --test test_rhtsdg_xndescent hierarchy_build_trace_reports_zero_layer_vector_copy_bytes -- --nocapture`
Expected: FAIL with missing trace field or nonzero copy count

- [ ] **Step 3: Replace copied layer vectors with global-buffer subset access**

```rust
struct NodeSubsetView<'a> {
    dim: usize,
    vectors: &'a [f32],
    nodes: &'a [u32],
}

impl<'a> NodeSubsetView<'a> {
    fn vector(&self, local_idx: usize) -> &'a [f32] {
        let global = self.nodes[local_idx] as usize;
        let start = global * self.dim;
        &self.vectors[start..start + self.dim]
    }
}
```

- [ ] **Step 4: Re-run build-side tests and the representative local lane**

Run: `cargo test --test test_rhtsdg_tsdg --test test_rhtsdg_xndescent -- --nocapture`
Expected: PASS

Run: `cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128`
Expected: PASS with unchanged recall and improved build time

- [ ] **Step 5: Commit only if the local lane improves**

```bash
git add src/faiss/rhtsdg/search.rs src/faiss/rhtsdg/xndescent.rs src/faiss/rhtsdg/tsdg.rs tests/test_rhtsdg_xndescent.rs tests/test_rhtsdg_tsdg.rs task-progress.md
git commit -m "perf(rhtsdg): remove hierarchy vector copies"
```

### Task 4: Replace clone-heavy neighborhood candidate extraction with scratch-backed snapshots

**Files:**
- Modify: `src/faiss/rhtsdg/neighbor.rs`
- Modify: `src/faiss/rhtsdg/xndescent.rs`
- Modify: `tests/test_rhtsdg_xndescent.rs`

- [ ] **Step 1: Write the failing candidate-extraction trace tests**

```rust
#[test]
fn rebuild_samples_trace_reports_no_pool_clone_per_iteration() {
    let trace = sampled_join_trace_fixture();
    assert_eq!(trace.pool_clone_count, 0);
}

#[test]
fn join_candidate_lists_remain_distance_sorted_and_deduped() {
    let fixture = sampled_join_fixture();
    assert_eq!(fixture.join_candidates(), vec![1, 2, 4]);
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `cargo test --test test_rhtsdg_xndescent rebuild_samples_trace_reports_no_pool_clone_per_iteration -- --nocapture`
Expected: FAIL with missing trace field or nonzero clone count

- [ ] **Step 3: Move candidate extraction onto reusable scratch buffers**

```rust
struct NeighborhoodScratch {
    ordered_ids: Vec<u32>,
    ordered_status: Vec<NeighborStatus>,
}

fn rebuild_samples_into(&self, scratch: &mut NeighborhoodScratch, cfg: &XNDescentConfig) {
    // reuse scratch instead of clone + sort of the whole pool
}
```

- [ ] **Step 4: Re-run XNDescent tests and the representative local lane**

Run: `cargo test --test test_rhtsdg_xndescent -- --nocapture`
Expected: PASS

Run: `cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128`
Expected: PASS with unchanged recall and improved build time

- [ ] **Step 5: Commit only if the local lane improves**

```bash
git add src/faiss/rhtsdg/neighbor.rs src/faiss/rhtsdg/xndescent.rs tests/test_rhtsdg_xndescent.rs task-progress.md
git commit -m "perf(rhtsdg): reduce xndescent neighborhood copy cost"
```

### Task 5: Reduce TSDG reverse-edge and occurrence-filter overhead without changing graph quality

**Files:**
- Modify: `src/faiss/rhtsdg/tsdg.rs`
- Modify: `tests/test_rhtsdg_tsdg.rs`

- [ ] **Step 1: Write the failing TSDG cost-audit tests**

```rust
#[test]
fn reverse_edge_trace_caps_stage2_candidate_rechecks() {
    let trace = tsdg_trace_fixture();
    assert!(trace.stage2_rechecks <= trace.reverse_edges_collected);
}

#[test]
fn occurrence_filter_keeps_existing_deterministic_order_on_ties() {
    let kept = tied_occurrence_fixture();
    assert_eq!(kept[0], vec![1, 3, 4]);
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `cargo test --test test_rhtsdg_tsdg reverse_edge_trace_caps_stage2_candidate_rechecks -- --nocapture`
Expected: FAIL with missing trace field or mismatched tie ordering

- [ ] **Step 3: Pre-size reverse-edge buffers and reuse occurrence scratch**

```rust
let mut reverse = vec![Vec::with_capacity(max_reverse_hint); base_graph.len()];
let mut occ_scratch = vec![0u32; max_k];
```

- [ ] **Step 4: Re-run TSDG tests and the representative local lane**

Run: `cargo test --test test_rhtsdg_tsdg -- --nocapture`
Expected: PASS

Run: `cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128`
Expected: PASS with unchanged recall and improved build time

- [ ] **Step 5: Commit only if the local lane improves**

```bash
git add src/faiss/rhtsdg/tsdg.rs tests/test_rhtsdg_tsdg.rs task-progress.md
git commit -m "perf(rhtsdg): reduce tsdg reverse-edge overhead"
```

## Chunk 3: Search-Side Optimization Screens

### Task 6: Cache result-threshold state and avoid repeated heap worst-distance lookups

**Files:**
- Modify: `src/faiss/rhtsdg/search.rs`
- Modify: `tests/test_rhtsdg_screen.rs`

- [ ] **Step 1: Write the failing threshold-cache tests**

```rust
#[test]
fn search_trace_reports_threshold_cache_hits_after_ef_is_full() {
    let trace = saturated_search_trace_fixture();
    assert!(trace.threshold_cache_hits > 0);
}

#[test]
fn search_results_remain_sorted_after_threshold_cache_updates() {
    let trace = sorted_search_trace_fixture();
    assert_eq!(trace.results, vec![1, 2, 3, 4]);
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `cargo test --test test_rhtsdg_screen search_trace_reports_threshold_cache_hits_after_ef_is_full -- --nocapture`
Expected: FAIL with missing trace field or zero cache hits

- [ ] **Step 3: Add an explicit layer-search state object**

```rust
struct LayerSearchState {
    frontier: BinaryHeap<FrontierEntry>,
    results: BinaryHeap<ResultEntry>,
    worst_result_dist: f32,
}
```

- [ ] **Step 4: Re-run search tests and the representative local lane**

Run: `cargo test --test test_rhtsdg_screen -- --nocapture`
Expected: PASS

Run: `cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128`
Expected: PASS with unchanged recall and improved search time

- [ ] **Step 5: Commit only if the local lane improves**

```bash
git add src/faiss/rhtsdg/search.rs tests/test_rhtsdg_screen.rs task-progress.md
git commit -m "perf(rhtsdg): cache layer search result threshold"
```

### Task 7: Optimize layer-0 frontier/result maintenance without changing traversal semantics

**Files:**
- Modify: `src/faiss/rhtsdg/search.rs`
- Modify: `tests/test_rhtsdg_screen.rs`
- Modify: `benches/rhtsdg_bench.rs`

- [ ] **Step 1: Write the failing layer-0 maintenance tests**

```rust
#[test]
fn layer0_trace_reports_fewer_result_heap_mutations_than_neighbors_seen() {
    let trace = saturated_search_trace_fixture();
    assert!(trace.result_heap_mutations < trace.visited);
}

#[test]
fn layer0_search_keeps_pruning_behavior_on_dominated_neighbors() {
    let trace = dominated_neighbor_fixture();
    assert_eq!(trace.frontier_pops, 2);
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `cargo test --test test_rhtsdg_screen layer0_trace_reports_fewer_result_heap_mutations_than_neighbors_seen -- --nocapture`
Expected: FAIL with missing trace field or unchanged mutation count

- [ ] **Step 3: Move result acceptance into a bounded helper with cached worst distance**

```rust
#[inline]
fn try_accept_result(state: &mut LayerSearchState, bitset: Option<&BitsetView>, entry: ResultEntry, ef: usize) {
    // update heap and cached threshold together
}
```

- [ ] **Step 4: Re-run search tests, bench lane, and representative local lane**

Run: `cargo test --test test_rhtsdg_screen -- --nocapture`
Expected: PASS

Run: `cargo bench --bench rhtsdg_bench -- --noplot`
Expected: PASS with no regression in the synthetic bench lane

Run: `cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128`
Expected: PASS with unchanged recall and improved search time

- [ ] **Step 5: Commit only if the local lane improves**

```bash
git add src/faiss/rhtsdg/search.rs tests/test_rhtsdg_screen.rs benches/rhtsdg_bench.rs task-progress.md
git commit -m "perf(rhtsdg): reduce layer0 frontier maintenance cost"
```

## Chunk 4: Promotion, Authority, and Durable Closure

### Task 8: Promote only a surviving local bundle to authority and document the verdict

**Files:**
- Modify: `task-progress.md`
- Modify: `feature-list.json`
- Modify: `RELEASE_NOTES.md`

- [ ] **Step 1: Verify a local bundle actually beats the current trusted lane**

Run: `cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128`
Expected: PASS and a result better than session 241 on the chosen target while keeping `recall@10 >= 0.9965`

- [ ] **Step 2: Stop immediately if no local bundle clears the gate**

Run: none
Expected: `task-progress.md` records `screen_result=needs_more_local`; `feature-list.json` and `RELEASE_NOTES.md` stay unchanged

- [ ] **Step 3: Sync to the authority machine only after a local promotion**

Run: `bash init.sh`
Expected: PASS with resolved remote config output

- [ ] **Step 4: Replay the authority verification chain**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-rhtsdg KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-rhtsdg bash scripts/remote/test.sh --command "cargo test --release --test test_rhtsdg_screen screen_rhtsdg_recall_gate_on_synthetic_fixture -- --ignored --nocapture"`
Expected: PASS with `recall_at_10=1.0`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-rhtsdg KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-rhtsdg bash scripts/remote/test.sh --command "cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128"`
Expected: PASS with improved authority `build_s` and/or `search_s` and unchanged recall

- [ ] **Step 5: Update durable records and commit the promoted bundle**

```bash
git add task-progress.md feature-list.json RELEASE_NOTES.md src/faiss/rhtsdg/*.rs tests/test_rhtsdg_*.rs benches/rhtsdg_bench.rs examples/rhtsdg_vs_hnsw.rs
git commit -m "perf(rhtsdg): improve build and search hot paths"
```

## Execution Notes

- Run one hypothesis at a time. Do not stack multiple build and search changes before measuring the local lane.
- Every failed local hypothesis must be reverted before the next one begins.
- Do not claim success from local results alone. The authoritative verdict remains the remote x86 lane.
- If a local bundle improves build but regresses search, or improves search but regresses recall, record it as `needs_more_local` and stop promotion.

