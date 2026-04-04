# RHTSDG Rust Screen and Core Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Rust-native RHTSDG implementation to `knowhere-rs` in a way that first proves local graph-quality viability through `screen`, then promotes only the Rust-core index surface into tracked authority work.

**Architecture:** Keep the implementation local to `src/faiss/rhtsdg/*` instead of introducing a new top-level `graph/` package. Reuse the existing `IndexConfig` / `IndexParams` / `Index` trait / `visited_pool` contracts, and gate the work behind the repository's `screen -> authority -> durable closure` workflow instead of opening a full production lane up front.

**Tech Stack:** Rust 2021, `rayon`, `parking_lot`, existing `Dataset` / `BitsetView` / `Index` traits, local cargo test/bench for `screen`, remote `scripts/remote/*` wrappers for authority verification

---

## Scope Notes

- This plan intentionally excludes `src/ffi.rs`, `src/python/mod.rs`, and `src/jni/mod.rs`.
- This plan intentionally excludes shortcut-graph work until base RHTSDG graph quality is proven.
- This plan intentionally excludes any native-parity or leadership claim until remote x86 artifacts exist.
- If the local `screen` result is not `promote`, stop after Chunk 1 and do not modify `feature-list.json`.

## Planned File Map

**Create**
- `src/faiss/rhtsdg/mod.rs`: main `RhtsdgIndex` type, `Index` trait implementation, persistence envelope wiring
- `src/faiss/rhtsdg/config.rs`: adapter from `IndexConfig` / `IndexParams` into internal `RhtsdgConfig`
- `src/faiss/rhtsdg/neighbor.rs`: safe neighborhood state, dedupe, bounded insert, new/old sampling helpers
- `src/faiss/rhtsdg/xndescent.rs`: `XNDescentBuilder` with local-join iteration and convergence accounting
- `src/faiss/rhtsdg/tsdg.rs`: Stage 1 alpha pruning, reverse-edge collection, Stage 2 occurrence filtering
- `src/faiss/rhtsdg/search.rs`: layer search implementation, private ordered candidate/results pools, visited-pool integration
- `tests/test_rhtsdg_tsdg.rs`: deterministic diversification fixtures
- `tests/test_rhtsdg_xndescent.rs`: deterministic XNDescent neighborhood/update fixtures
- `tests/test_rhtsdg_index_trait.rs`: Rust `Index` trait, persistence, bitset, and iterator coverage
- `tests/test_rhtsdg_screen.rs`: local `screen` correctness and recall-gate diagnostics
- `benches/rhtsdg_bench.rs`: post-promotion Criterion benchmark
- `examples/rhtsdg_vs_hnsw.rs`: post-promotion comparison driver

**Modify**
- `src/faiss/mod.rs`: module wiring and public export for `RhtsdgIndex`
- `src/lib.rs`: top-level re-export
- `src/api/index.rs`: `IndexType::Rhtsdg` and `IndexParams` fields prefixed with `rhtsdg_*`
- `src/api/legal_matrix.rs`: legal combinations and mmap declaration
- `src/codec/index.rs`: config codec id for `RHTSDG`
- `task-progress.md`: record `screen` hypothesis, commands, result, and next step
- `feature-list.json`: add tracked RHTSDG features only after `screen_result=promote`
- `RELEASE_NOTES.md`: tracked-work summary after authority verification

**Do Not Create In First Pass**
- `src/graph/*`
- `src/search/linear_pool.rs`
- any new FFI/JNI/Python entrypoint

## Chunk 1: Local Screen and Algorithm Proof

### Task 1: Lock TSDG behavior with deterministic fixtures

**Files:**
- Create: `tests/test_rhtsdg_tsdg.rs`
- Create: `src/faiss/rhtsdg/tsdg.rs`
- Create: `src/faiss/rhtsdg/neighbor.rs`

- [ ] **Step 1: Write the failing TSDG tests**

```rust
#[test]
fn alpha_pruning_drops_center_sorted_occluded_neighbor() {
    let fixture = DistanceFixture::line_with_off_axis_escape();
    let kept = fixture.run_stage1(1.2);
    assert_eq!(kept[0], vec![1, 3]);
}

#[test]
fn occurrence_filter_adds_reverse_edge_only_below_threshold() {
    let fixture = DistanceFixture::reverse_edge_gate();
    let kept = fixture.run_stage2(1.2, 1);
    assert_eq!(kept[0], vec![1, 4]);
}
```

- [ ] **Step 2: Run the new test target and verify it fails**

Run: `cargo test --test test_rhtsdg_tsdg -- --nocapture`
Expected: FAIL with unresolved `rhtsdg` module items or failing assertions

- [ ] **Step 3: Implement Stage 1 and Stage 2 with center-distance ordering**

```rust
pub(crate) fn sort_neighbors_by_center_distance(
    center: usize,
    neighbors: &[u32],
    distance: impl Fn(usize, usize) -> f32,
) -> Vec<(u32, f32)> {
    let mut ordered: Vec<(u32, f32)> = neighbors
        .iter()
        .map(|&id| (id, distance(center, id as usize)))
        .collect();
    ordered.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    ordered
}
```

- [ ] **Step 4: Re-run the TSDG tests and verify they pass**

Run: `cargo test --test test_rhtsdg_tsdg -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit the TSDG fixture lock**

```bash
git add tests/test_rhtsdg_tsdg.rs src/faiss/rhtsdg/tsdg.rs src/faiss/rhtsdg/neighbor.rs
git commit -m "test(rhtsdg): lock tsdg diversification fixtures"
```

### Task 2: Build a safe XNDescent neighborhood model before parallel iteration

**Files:**
- Modify: `src/faiss/rhtsdg/neighbor.rs`
- Create: `src/faiss/rhtsdg/xndescent.rs`
- Create: `tests/test_rhtsdg_xndescent.rs`

- [ ] **Step 1: Write the failing neighborhood and local-join tests**

```rust
#[test]
fn insert_neighbor_dedupes_and_keeps_best_k() {
    let mut n = Neighborhood::new(3);
    assert!(n.insert(7, 0.3, NeighborStatus::New));
    assert!(!n.insert(7, 0.5, NeighborStatus::New));
    assert!(n.insert(8, 0.2, NeighborStatus::Old));
    assert_eq!(n.snapshot().len(), 2);
}

#[test]
fn local_join_updates_both_endpoints_symmetrically() {
    let mut builder = ScreenXnDescentFixture::new();
    let updates = builder.local_join_once();
    assert!(updates > 0);
    assert!(builder.has_edge(1, 2));
    assert!(builder.has_edge(2, 1));
}
```

- [ ] **Step 2: Run the XNDescent tests and verify they fail**

Run: `cargo test --test test_rhtsdg_xndescent -- --nocapture`
Expected: FAIL with missing `Neighborhood` / `XNDescentBuilder` functionality

- [ ] **Step 3: Implement safe interior mutability for neighborhoods**

```rust
pub struct Neighborhood {
    inner: parking_lot::Mutex<NeighborhoodState>,
}

struct NeighborhoodState {
    pool: Vec<Neighbor>,
    nn_new: Vec<u32>,
    nn_old: Vec<u32>,
    rnn_new: Vec<u32>,
    rnn_old: Vec<u32>,
}
```

- [ ] **Step 4: Implement `update_sample_neighbors()` and `local_join()` against snapshots**

Run: `cargo test --test test_rhtsdg_xndescent -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit the safe XNDescent core**

```bash
git add src/faiss/rhtsdg/neighbor.rs src/faiss/rhtsdg/xndescent.rs tests/test_rhtsdg_xndescent.rs
git commit -m "feat(rhtsdg): add safe xndescent neighborhood core"
```

### Task 3: Add a single-layer search prototype and make a local `screen` decision

**Files:**
- Create: `src/faiss/rhtsdg/search.rs`
- Create: `src/faiss/rhtsdg/mod.rs`
- Create: `tests/test_rhtsdg_screen.rs`
- Modify: `task-progress.md`

- [ ] **Step 1: Write the failing search and recall-gate tests**

```rust
#[test]
fn search_matches_bruteforce_top1_on_small_grid() {
    let (index, queries, truth) = small_grid_fixture();
    let got = index.search_batch_for_test(&queries, 1, 16);
    assert_eq!(got, truth);
}

#[test]
#[ignore]
fn screen_rhtsdg_recall_gate_on_sift_fixture() {
    let summary = run_local_screen_fixture();
    assert!(summary.recall_at_10 >= 0.95, "recall={}", summary.recall_at_10);
}
```

- [ ] **Step 2: Run the search test target and verify it fails**

Run: `cargo test --test test_rhtsdg_screen search_matches_bruteforce_top1_on_small_grid -- --nocapture`
Expected: FAIL with missing `RhtsdgIndex` search path

- [ ] **Step 3: Implement private ordered candidate/results pools and reuse `visited_pool`**

```rust
pub(crate) fn search_layer(
    &self,
    query: &[f32],
    entry_points: &[u32],
    ef: usize,
    layer: usize,
) -> Vec<(u32, f32)> {
    crate::search::with_visited(self.len(), |visited| {
        // ordered frontier + ordered results, not a dedupe-free linear bag
        self.search_layer_inner(query, entry_points, ef, layer, visited)
    })
}
```

- [ ] **Step 4: Run the local screen commands**

Run: `cargo test --test test_rhtsdg_tsdg -- --nocapture`
Expected: PASS

Run: `cargo test --test test_rhtsdg_xndescent -- --nocapture`
Expected: PASS

Run: `cargo test --test test_rhtsdg_screen search_matches_bruteforce_top1_on_small_grid -- --nocapture`
Expected: PASS

Run: `cargo test --release --test test_rhtsdg_screen screen_rhtsdg_recall_gate_on_sift_fixture -- --ignored --nocapture`
Expected: either PASS with a candidate `screen_result=promote`, or FAIL with a reproducible local rejection signal

- [ ] **Step 5: Record the `screen` result in `task-progress.md`**

Capture:
- hypothesis: "offline XNDescent + TSDG yields higher-quality base graph than current HNSW-like online pruning"
- commands run
- local recall/build observations
- decision: `screen_result=promote` or `screen_result=reject` or `screen_result=needs_more_local`

- [ ] **Step 6: Commit the screen outcome**

```bash
git add src/faiss/rhtsdg/mod.rs src/faiss/rhtsdg/search.rs tests/test_rhtsdg_screen.rs task-progress.md
git commit -m "feat(rhtsdg): add local screen prototype and decision log"
```

## Chunk 2: Promote to Rust-Core Index Surface Only If Screen Passed

### Task 4: Add Rust API/config plumbing without widening external ABI

**Files:**
- Modify: `src/api/index.rs`
- Modify: `src/api/legal_matrix.rs`
- Modify: `src/faiss/mod.rs`
- Modify: `src/lib.rs`
- Create: `src/faiss/rhtsdg/config.rs`
- Create: `tests/test_rhtsdg_index_trait.rs`

- [ ] **Step 1: Write the failing config parse/validate tests**

```rust
#[test]
fn rhtsdg_index_type_parses_from_string() {
    assert_eq!("rhtsdg".parse::<IndexType>().unwrap(), IndexType::Rhtsdg);
}

#[test]
fn rhtsdg_float_l2_config_is_legal() {
    let cfg = IndexConfig::new(IndexType::Rhtsdg, MetricType::L2, 128);
    assert!(cfg.validate().is_ok());
}
```

- [ ] **Step 2: Run the targeted config tests and verify they fail**

Run: `cargo test --lib rhtsdg_index_type_parses_from_string rhtsdg_float_l2_config_is_legal -- --nocapture`
Expected: FAIL because `IndexType::Rhtsdg` does not exist yet

- [ ] **Step 3: Add `IndexType::Rhtsdg` and prefixed `IndexParams` fields**

Add fields to `IndexParams`:
- `rhtsdg_alpha: Option<f32>`
- `rhtsdg_occ_threshold: Option<u32>`
- `rhtsdg_knn_k: Option<usize>`
- `rhtsdg_sample_count: Option<usize>`
- `rhtsdg_iter_count: Option<usize>`
- `rhtsdg_reverse_count: Option<usize>`
- `rhtsdg_use_shortcut: Option<bool>`

Do not add new top-level `RhtsdgConfig` to public API; adapt from `IndexConfig` in `src/faiss/rhtsdg/config.rs`.

- [ ] **Step 4: Export the Rust index type and validate legality**

Run: `cargo test --lib rhtsdg_index_type_parses_from_string rhtsdg_float_l2_config_is_legal -- --nocapture`
Expected: PASS

- [ ] **Step 5: Commit the Rust-only config surface**

```bash
git add src/api/index.rs src/api/legal_matrix.rs src/faiss/mod.rs src/lib.rs src/faiss/rhtsdg/config.rs tests/test_rhtsdg_index_trait.rs
git commit -m "feat(rhtsdg): add rust-core config and module wiring"
```

### Task 5: Implement the full Rust `Index` trait and persistence contract

**Files:**
- Modify: `src/faiss/rhtsdg/mod.rs`
- Modify: `src/faiss/rhtsdg/search.rs`
- Modify: `src/codec/index.rs`
- Modify: `tests/test_rhtsdg_index_trait.rs`

- [ ] **Step 1: Write the failing trait-contract tests**

```rust
#[test]
fn rhtsdg_index_round_trips_save_and_load() {
    let (mut index, data, query) = persistence_fixture();
    index.train(&data).unwrap();
    index.add(&data).unwrap();
    let before = index.search(&query, 3).unwrap();
    let path = tempfile::NamedTempFile::new().unwrap();
    index.save(path.path().to_str().unwrap()).unwrap();

    let mut loaded = RhtsdgIndex::new(&index.config().clone()).unwrap();
    loaded.load(path.path().to_str().unwrap()).unwrap();
    let after = loaded.search(&query, 3).unwrap();
    assert_eq!(before.ids, after.ids);
}
```

- [ ] **Step 2: Run the trait-contract tests and verify they fail**

Run: `cargo test --test test_rhtsdg_index_trait -- --nocapture`
Expected: FAIL with missing `Index` trait behavior, persistence, or iterator support

- [ ] **Step 3: Implement the Rust index contract**

Implement:
- `pub fn new(config: &IndexConfig) -> Result<Self>`
- `train(&mut self, dataset: &Dataset)`
- `add(&mut self, dataset: &Dataset)`
- `search(&self, query: &Dataset, top_k: usize)`
- `search_with_bitset(...)`
- `save(&self, path: &str)` / `load(&mut self, path: &str)`
- `get_vector_by_ids(&self, ids: &[i64])`
- `create_ann_iterator(...)`
- `has_raw_data() -> true`

Store only serializable state on disk; reconstruct RNGs, function pointers, and scratch/runtime helpers on load.

- [ ] **Step 4: Add a codec id and re-run the trait-contract tests**

Run: `cargo test --test test_rhtsdg_index_trait -- --nocapture`
Expected: PASS

- [ ] **Step 5: Run repository hygiene gates before commit**

Run: `cargo fmt --all -- --check`
Expected: PASS

Run: `cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS

- [ ] **Step 6: Commit the full Rust-core index**

```bash
git add src/faiss/rhtsdg/mod.rs src/faiss/rhtsdg/search.rs src/codec/index.rs tests/test_rhtsdg_index_trait.rs
git commit -m "feat(rhtsdg): implement rust index trait and persistence"
```

## Chunk 3: Tracked Work, Benchmarks, and Authority Verification

### Task 6: Promote the screen result into tracked work and add benchmark coverage

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Create: `benches/rhtsdg_bench.rs`
- Create: `examples/rhtsdg_vs_hnsw.rs`

- [ ] **Step 1: Add tracked RHTSDG features only if `screen_result=promote`**

Add at least:
- `rhtsdg-rust-core-index`
- `rhtsdg-remote-recall-benchmark`

Both features must include remote verification steps and dependency edges.

- [ ] **Step 2: Write the failing local comparison coverage**

```rust
fn bench_rhtsdg_build_and_search(c: &mut Criterion) {
    // compare build throughput and search latency against HNSW on the same fixture
}
```

- [ ] **Step 3: Run local prefilter verification**

Run: `cargo test --test test_rhtsdg_tsdg -- --nocapture`
Expected: PASS

Run: `cargo test --test test_rhtsdg_xndescent -- --nocapture`
Expected: PASS

Run: `cargo test --test test_rhtsdg_index_trait -- --nocapture`
Expected: PASS

Run: `cargo bench --bench rhtsdg_bench -- --noplot`
Expected: benchmark completes locally without panic

- [ ] **Step 4: Bootstrap authority before any tracked verdict**

Run: `bash init.sh`
Expected: remote config printed and sync succeeds

- [ ] **Step 5: Run authority verification for the tracked feature**

Run: `bash scripts/remote/test.sh --command "cargo test --test test_rhtsdg_index_trait -- --nocapture"`
Expected: PASS

Run: `bash scripts/remote/test.sh --command "cargo test --test test_rhtsdg_screen screen_rhtsdg_recall_gate_on_sift_fixture -- --ignored --nocapture"`
Expected: PASS if the local screen recall gate survives the authority machine

Run: `bash scripts/remote/test.sh --command "cargo run --release --example rhtsdg_vs_hnsw -- --dataset sift1m --top-k 10 --ef-search 128"`
Expected: comparison artifact prints build/search/recall summary for both indexes

- [ ] **Step 6: Validate tracked metadata and commit**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID - ... features`

```bash
git add feature-list.json task-progress.md RELEASE_NOTES.md benches/rhtsdg_bench.rs examples/rhtsdg_vs_hnsw.rs
git commit -m "feat(rhtsdg): add tracked benchmark and authority verification plan"
```

## Exit Criteria

- `Chunk 1` ends with exactly one recorded local decision in `task-progress.md`.
- `Chunk 2` is entered only if `Chunk 1` produced `screen_result=promote`.
- `Chunk 3` makes no performance or stop-go claim without remote x86 outputs.
- No change in this plan widens the C ABI or Python/JNI surface.
- `shortcut graph` remains disabled until the base RHTSDG build/search path is stable and recall-gated.

## Follow-Up Plan Required Later

Write a separate plan if any of the following become required:
- C ABI support in `src/ffi.rs`
- Python binding creation in `src/python/mod.rs`
- JNI registry support in `src/jni/mod.rs`
- native benchmark schema integration for durable RS-vs-native reporting
