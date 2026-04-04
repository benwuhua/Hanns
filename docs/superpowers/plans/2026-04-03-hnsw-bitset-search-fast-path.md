# HNSW Bitset Search Fast Path — Milvus QPS Recovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the 40%+ QPS gap between knowhere-rs and native knowhere in Milvus VectorDBBench by optimizing the bitset search hot path in HNSW.

**Architecture:** The Milvus FFI always calls `search_with_bitset`, which dispatches to `search_single_with_bitset` → `search_layer_idx_shared` (the slow generic path). The L2 unfiltered path has a dedicated fast searcher (`search_layer_idx_l2_ordered_pool_fast`) using `layer0_slab`, SIMD batch-4, and `Layer0OrderedPool` data structures — but none of this is used for bitset queries. We fix three independent bottlenecks: (1) TLS scratch reuse, (2) greedy upper-layer descent, (3) dedicated cosine+bitset fast path at layer 0.

**Tech Stack:** Rust, SIMD (via existing `simd::` module), `thread_local!` TLS, existing `Layer0Slab`/`Layer0OrderedPool` data structures.

**Key file:** `src/faiss/hnsw.rs`

---

## File Structure

All changes are in a single file:

- **Modify:** `src/faiss/hnsw.rs`
  - `search_single_with_bitset` — rewrite to use TLS scratch + greedy upper descent + fast layer0 dispatcher
  - New function `search_layer0_bitset_fast` — cosine/L2 + bitset fast path using slab + ordered pool
  - Tests — new tests for TLS reuse, bitset fast path correctness, slab+bitset interplay

No new files created. All changes localized to the existing HNSW module.

---

### Task 1: TLS Scratch Reuse for Bitset Search

The simplest, highest-confidence fix. `search_single_with_bitset` currently creates a new `SearchScratch` per query (2MB malloc/free for 500K nodes). Switch to the existing `HNSW_SEARCH_SCRATCH_TLS`.

**Files:**
- Modify: `src/faiss/hnsw.rs:6252-6328` (`search_single_with_bitset`)
- Test: `src/faiss/hnsw.rs` (add new test near existing scratch tests)

- [ ] **Step 1: Write the failing test**

Add this test after the existing `test_search_layer_idx_l2_heap_reuses_generic_heap_capacity_across_calls` test (~line 9971):

```rust
#[test]
fn test_search_single_with_bitset_reuses_tls_scratch() {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::Cosine,
        dim: 2,
        data_type: crate::api::DataType::Float,
        params: crate::api::IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(200),
            ..Default::default()
        },
    };
    let vectors: Vec<f32> = (0..200)
        .flat_map(|i| {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / 200.0;
            vec![angle.cos(), angle.sin()]
        })
        .collect();

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    let bitset = crate::bitset::Bitset::new(200);
    let bitset_view = bitset.as_view();

    let query = [1.0_f32, 0.0];

    // Run search twice — second call should NOT allocate new scratch
    let r1 = index.search_single_with_bitset(&query, 50, 10, &bitset_view);
    let r2 = index.search_single_with_bitset(&query, 50, 10, &bitset_view);

    // Results must be identical (deterministic with same graph)
    assert_eq!(r1.len(), r2.len(), "bitset search results must be deterministic across calls");
    for (a, b) in r1.iter().zip(r2.iter()) {
        assert_eq!(a.0, b.0, "IDs must match across calls");
        assert!((a.1 - b.1).abs() < 1e-6, "distances must match across calls");
    }

    // Verify TLS scratch has been sized (not empty)
    HNSW_SEARCH_SCRATCH_TLS.with(|cell| {
        let scratch = cell.borrow();
        assert!(
            scratch.visited_epoch.len() >= 200,
            "TLS scratch visited_epoch should have been sized to at least node count ({}) but was {}",
            200,
            scratch.visited_epoch.len()
        );
    });
}
```

- [ ] **Step 2: Run the test to confirm it fails**

Run: `cargo test test_search_single_with_bitset_reuses_tls_scratch -- --nocapture 2>&1 | tail -20`

Expected: FAIL — `search_single_with_bitset` is private and/or TLS scratch is empty because current code uses `SearchScratch::new()`.

Note: `search_single_with_bitset` is currently `fn` (private). The test is in the same module so visibility is fine. The failure should be that the TLS assertion fails (scratch.visited_epoch.len() == 0 because the current code never touches TLS).

- [ ] **Step 3: Implement TLS scratch reuse**

Replace the body of `search_single_with_bitset` (lines 6252-6328) with:

```rust
fn search_single_with_bitset(
    &self,
    query: &[f32],
    ef: usize,
    k: usize,
    bitset: &crate::bitset::BitsetView,
) -> Vec<(i64, f32)> {
    if self.ids.is_empty() || self.entry_point.is_none() {
        return vec![];
    }

    if self.metric_type == MetricType::Cosine && self.ids.len() <= ef.max(k).max(64) {
        return self
            .brute_force_search(query, k, |_id, idx| idx >= bitset.len() || !bitset.get(idx));
    }

    self.prepare_search_query_context(query);
    let result = HNSW_SEARCH_SCRATCH_TLS.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        self.search_single_with_bitset_scratch(query, ef, k, bitset, &mut scratch)
    });
    self.clear_search_query_context();
    result
}

fn search_single_with_bitset_scratch(
    &self,
    query: &[f32],
    ef: usize,
    k: usize,
    bitset: &crate::bitset::BitsetView,
    scratch: &mut SearchScratch,
) -> Vec<(i64, f32)> {
    let mut curr_ep_idx = self.get_idx_from_id_fast(self.entry_point.unwrap());

    // Upper layer descent — greedy, no need for full search
    let mut curr_ep_dist = self.distance(query, curr_ep_idx);
    for level in (1..=self.max_level).rev() {
        let (next_idx, next_dist) = self.greedy_upper_layer_descent_idx_with_entry_dist(
            query,
            curr_ep_idx,
            level,
            curr_ep_dist,
        );
        curr_ep_idx = next_idx;
        curr_ep_dist = next_dist;
    }

    // Layer 0 search with bitset
    let results = self.search_layer_idx_with_bitset_scratch(
        query,
        curr_ep_idx,
        0,
        ef,
        bitset,
        scratch,
    );

    let mut final_results: Vec<(i64, f32)> = Vec::with_capacity(k);
    for (idx, dist) in results {
        let id = self.get_id_from_idx(idx);
        if self.is_deleted(id) {
            continue;
        }
        final_results.push((id, dist));
        if final_results.len() >= k {
            break;
        }
    }

    self.rerank_sq_results(query, &mut final_results);
    final_results
}
```

Key changes:
1. `SearchScratch::new()` replaced with `HNSW_SEARCH_SCRATCH_TLS.with(...)` — eliminates 2MB malloc/free per query
2. Upper layer descent uses `greedy_upper_layer_descent_idx_with_entry_dist` instead of full `search_layer_idx_shared` — removes BinaryHeap + profile overhead for upper layers
3. `prepare_search_query_context` and `clear_search_query_context` moved to outer wrapper for proper TLS norm management

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test test_search_single_with_bitset_reuses_tls_scratch -- --nocapture 2>&1 | tail -20`

Expected: PASS

- [ ] **Step 5: Run existing bitset-related tests to confirm no regression**

Run: `cargo test bitset -- --nocapture 2>&1 | tail -30`

Expected: All existing bitset tests pass.

- [ ] **Step 6: Run full HNSW test suite**

Run: `cargo test hnsw -- 2>&1 | tail -5`

Expected: All tests pass. If any fail, the change introduced a regression — investigate before proceeding.

- [ ] **Step 7: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): reuse TLS scratch + greedy upper descent in bitset search

Eliminates per-query SearchScratch allocation (~2MB for 500K nodes) in the
Milvus FFI search_with_bitset path. Replaces full search_layer_idx_shared
upper-layer descent with lightweight greedy descent.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: Dedicated Layer-0 Bitset Fast Path (Cosine + L2)

Add a `search_layer0_bitset_fast` function that uses `layer0_slab` + `ip_batch_4`/`l2_batch_4_ptrs` + `Layer0OrderedPool` for layer-0 bitset search. This is the highest-impact change for QPS.

**Files:**
- Modify: `src/faiss/hnsw.rs` — add new function + wire it into `search_single_with_bitset_scratch`
- Test: `src/faiss/hnsw.rs` — add correctness test comparing fast path vs shared path

- [ ] **Step 1: Write the failing test**

Add this test:

```rust
#[test]
fn test_search_layer0_bitset_fast_matches_shared_path() {
    // Build a cosine index with enough nodes to exercise slab path
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::Cosine,
        dim: 2,
        data_type: crate::api::DataType::Float,
        params: crate::api::IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(200),
            ..Default::default()
        },
    };
    let n = 500;
    let vectors: Vec<f32> = (0..n)
        .flat_map(|i| {
            let angle = (i as f32) * std::f32::consts::PI * 2.0 / (n as f32);
            vec![angle.cos(), angle.sin()]
        })
        .collect();

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    assert!(
        index.layer0_slab.is_enabled_for(index.node_info.len()),
        "slab must be enabled for this test to exercise the fast path"
    );

    // Create bitset that filters out nodes 0-49
    let mut bitset = crate::bitset::Bitset::new(n);
    for i in 0..50 {
        bitset.set(i);
    }
    let bitset_view = bitset.as_view();

    let query = [1.0_f32, 0.0];
    let ef = 50;
    let k = 10;

    // Reference: shared path
    index.prepare_search_query_context(&query);
    let mut scratch_ref = SearchScratch::new();
    let reference = index.search_layer_idx_with_bitset_scratch(
        &query, 0, 0, ef, &bitset_view, &mut scratch_ref,
    );
    index.clear_search_query_context();

    // Fast path
    index.prepare_search_query_context(&query);
    let mut scratch_fast = SearchScratch::new();
    let fast = index.search_layer0_bitset_fast(
        &query, 0, ef, &bitset_view, &mut scratch_fast,
    );
    index.clear_search_query_context();

    // Same set of result indices (order might differ for equal distances)
    let ref_ids: std::collections::HashSet<usize> = reference.iter().map(|r| r.0).collect();
    let fast_ids: std::collections::HashSet<usize> = fast.iter().map(|r| r.0).collect();
    assert_eq!(
        ref_ids, fast_ids,
        "fast bitset path must return same result set as shared path"
    );

    // No filtered nodes in results
    for (idx, _dist) in &fast {
        assert!(
            *idx >= 50,
            "fast path result idx {} should have been filtered by bitset",
            idx,
        );
    }
}
```

- [ ] **Step 2: Run test to confirm it fails**

Run: `cargo test test_search_layer0_bitset_fast_matches_shared_path -- --nocapture 2>&1 | tail -20`

Expected: FAIL — `search_layer0_bitset_fast` does not exist yet.

- [ ] **Step 3: Implement `search_layer0_bitset_fast`**

Add this function to `HnswIndex` impl (near `search_layer_idx_l2_ordered_pool_fast` around line 5152):

```rust
/// Layer-0 bitset search using slab layout + ordered pool.
///
/// Falls back to `search_layer_idx_with_bitset_scratch` when slab is not available.
fn search_layer0_bitset_fast(
    &self,
    query: &[f32],
    entry_idx: usize,
    ef: usize,
    bitset: &crate::bitset::BitsetView,
    scratch: &mut SearchScratch,
) -> Vec<(usize, f32)> {
    let num_nodes = self.node_info.len();

    // Require slab + flat_graph for fast path
    if !self.layer0_slab.is_enabled_for(num_nodes)
        || !self.layer0_flat_graph.is_enabled_for(num_nodes)
    {
        return self.search_layer_idx_with_bitset_scratch(
            query, entry_idx, 0, ef, bitset, scratch,
        );
    }

    let is_l2 = self.metric_type == MetricType::L2;
    let query_ptr = query.as_ptr();
    let base_ptr = self.vectors.as_ptr();

    scratch.prepare(num_nodes);
    scratch.prepare_layer0_pools(ef);

    let entry_dist = if is_l2 {
        unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, entry_idx) }
    } else {
        self.distance(query, entry_idx)
    };

    let entry = Layer0PoolEntry {
        idx: entry_idx,
        dist: entry_dist,
    };
    scratch.layer0_frontier.push(entry);
    if entry_idx >= bitset.len() || !bitset.get(entry_idx) {
        scratch.layer0_results.insert(entry, ef);
    }
    scratch.mark_visited(entry_idx);

    loop {
        let Some(candidate) = scratch.layer0_frontier.pop_best() else {
            break;
        };

        if scratch.layer0_results.len() >= ef
            && scratch
                .layer0_results
                .worst_dist()
                .is_some_and(|worst_dist| candidate.dist >= worst_dist)
        {
            break;
        }

        let neighbors = self.layer0_slab.neighbors_for(candidate.idx);
        let mut batch_indices = [0usize; 4];
        let mut batch_len = 0usize;

        for &nbr_u32 in neighbors {
            let nbr_idx = nbr_u32 as usize;
            if nbr_idx >= num_nodes {
                continue;
            }
            if !unsafe { scratch.mark_visited_unchecked(nbr_idx) } {
                continue;
            }

            batch_indices[batch_len] = nbr_idx;
            batch_len += 1;

            if batch_len == 4 {
                let distances = if is_l2 {
                    unsafe {
                        [
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[0]),
                                self.dim,
                            ),
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[1]),
                                self.dim,
                            ),
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[2]),
                                self.dim,
                            ),
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[3]),
                                self.dim,
                            ),
                        ]
                    }
                } else {
                    // Cosine/IP: use batch-4 dispatch (handles norm)
                    match Self::distance_to_4_idxs_dispatch(
                        self,
                        query,
                        batch_indices,
                        query_ptr,
                        base_ptr,
                    ) {
                        Some(d) => d,
                        None => {
                            // fallback: scalar one-by-one
                            [
                                self.distance(query, batch_indices[0]),
                                self.distance(query, batch_indices[1]),
                                self.distance(query, batch_indices[2]),
                                self.distance(query, batch_indices[3]),
                            ]
                        }
                    }
                };

                for i in 0..4 {
                    let nbr_idx = batch_indices[i];
                    let is_filtered =
                        nbr_idx < bitset.len() && bitset.get(nbr_idx);
                    let entry = Layer0PoolEntry {
                        idx: nbr_idx,
                        dist: distances[i],
                    };
                    // Always add to frontier (filtered nodes help traverse the graph)
                    scratch.layer0_frontier.push(entry);
                    // Only add to results if not filtered
                    if !is_filtered {
                        scratch.layer0_results.insert(entry, ef);
                    }
                }
                batch_len = 0;
            }
        }

        // Flush remaining batch
        for i in 0..batch_len {
            let nbr_idx = batch_indices[i];
            let nbr_dist = if is_l2 {
                unsafe { self.l2_distance_to_idx_ptr(query_ptr, base_ptr, nbr_idx) }
            } else {
                self.distance(query, nbr_idx)
            };
            let is_filtered = nbr_idx < bitset.len() && bitset.get(nbr_idx);
            let entry = Layer0PoolEntry {
                idx: nbr_idx,
                dist: nbr_dist,
            };
            scratch.layer0_frontier.push(entry);
            if !is_filtered {
                scratch.layer0_results.insert(entry, ef);
            }
        }
    }

    scratch.layer0_results.drain_sorted()
}
```

- [ ] **Step 4: Wire the fast path into `search_single_with_bitset_scratch`**

In `search_single_with_bitset_scratch` (created in Task 1), replace the layer-0 search call:

Replace:
```rust
    // Layer 0 search with bitset
    let results = self.search_layer_idx_with_bitset_scratch(
        query,
        curr_ep_idx,
        0,
        ef,
        bitset,
        scratch,
    );
```

With:
```rust
    // Layer 0 search with bitset — use fast path when slab available
    let results = self.search_layer0_bitset_fast(
        query,
        curr_ep_idx,
        ef,
        bitset,
        scratch,
    );
```

- [ ] **Step 5: Run new test**

Run: `cargo test test_search_layer0_bitset_fast_matches_shared_path -- --nocapture 2>&1 | tail -20`

Expected: PASS

- [ ] **Step 6: Run full HNSW test suite**

Run: `cargo test hnsw -- 2>&1 | tail -5`

Expected: All pass.

- [ ] **Step 7: Run FFI bitset tests**

Run: `cargo test bitset -- 2>&1 | tail -10`

Expected: All pass.

- [ ] **Step 8: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): add dedicated layer0 bitset fast path with slab + ordered pool

Introduces search_layer0_bitset_fast() using layer0_slab for contiguous
neighbor+vector access and Layer0OrderedPool for frontier/results. Supports
both L2 and Cosine metrics. Falls back to search_layer_idx_shared when slab
is not available. Bitset-filtered nodes still participate in graph traversal
but are excluded from results.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: L2 Slab Batch-4 SIMD in Bitset Fast Path

Task 2 uses scalar L2 per-vector distances via `l2_distance_sq_ptr_kernel`. Replace with true SIMD batch-4 from slab pointers (matching the existing L2 fast path at line 5236).

**Files:**
- Modify: `src/faiss/hnsw.rs` — update L2 branch in `search_layer0_bitset_fast`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_search_layer0_bitset_fast_l2_uses_slab_batch4() {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: crate::api::DataType::Float,
        params: crate::api::IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(200),
            ..Default::default()
        },
    };
    let n = 200;
    let mut rng = crate::faiss::test_utils::seeded_rng(42);
    let vectors: Vec<f32> = (0..n * 4)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    assert!(index.layer0_slab.is_enabled_for(n));

    let mut bitset = crate::bitset::Bitset::new(n);
    for i in 0..20 {
        bitset.set(i);
    }
    let bitset_view = bitset.as_view();

    let query = [0.5_f32, 0.3, -0.1, 0.7];
    let ef = 50;

    // Reference via shared path
    let mut scratch_ref = SearchScratch::new();
    let reference = index.search_layer_idx_with_bitset_scratch(
        &query, 0, 0, ef, &bitset_view, &mut scratch_ref,
    );

    // Fast path
    let mut scratch_fast = SearchScratch::new();
    let fast = index.search_layer0_bitset_fast(
        &query, 0, ef, &bitset_view, &mut scratch_fast,
    );

    let ref_ids: std::collections::HashSet<usize> = reference.iter().map(|r| r.0).collect();
    let fast_ids: std::collections::HashSet<usize> = fast.iter().map(|r| r.0).collect();
    assert_eq!(ref_ids, fast_ids, "L2 slab batch-4 bitset results must match shared path");
}
```

- [ ] **Step 2: Run test to confirm it passes with scalar fallback**

Run: `cargo test test_search_layer0_bitset_fast_l2_uses_slab_batch4 -- --nocapture 2>&1 | tail -20`

Expected: PASS (scalar L2 from Task 2 produces correct results; we optimize in next step).

- [ ] **Step 3: Replace scalar L2 with slab SIMD batch-4**

In `search_layer0_bitset_fast`, replace the L2 branch inside the `if batch_len == 4` block:

Replace:
```rust
                let distances = if is_l2 {
                    unsafe {
                        [
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[0]),
                                self.dim,
                            ),
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[1]),
                                self.dim,
                            ),
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[2]),
                                self.dim,
                            ),
                            (self.l2_distance_sq_ptr_kernel)(
                                query_ptr,
                                self.layer0_slab.vector_ptr_for(batch_indices[3]),
                                self.dim,
                            ),
                        ]
                    }
```

With:
```rust
                let distances = if is_l2 {
                    unsafe {
                        simd::l2_batch_4_ptrs(
                            query_ptr,
                            self.layer0_slab.vector_ptr_for(batch_indices[0]),
                            self.layer0_slab.vector_ptr_for(batch_indices[1]),
                            self.layer0_slab.vector_ptr_for(batch_indices[2]),
                            self.layer0_slab.vector_ptr_for(batch_indices[3]),
                            self.dim,
                        )
                    }
```

- [ ] **Step 4: Run test to confirm still passes**

Run: `cargo test test_search_layer0_bitset_fast_l2_uses_slab_batch4 -- --nocapture 2>&1 | tail -20`

Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `cargo test hnsw -- 2>&1 | tail -5`

Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): use SIMD l2_batch_4_ptrs from slab in bitset fast path

Replaces per-vector L2 kernel calls with true SIMD batch-4 distance
computation using slab vector pointers. Matches the optimization already
used in search_layer_idx_l2_ordered_pool_fast.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: Cosine Slab Batch-4 in Bitset Fast Path

The Cosine branch in Task 2 falls back to `distance_to_4_idxs_dispatch` which reads vectors from `self.vectors` (not slab). Add a slab-native cosine batch-4 path using `ip_batch_4` on slab vector pointers.

**Files:**
- Modify: `src/faiss/hnsw.rs` — update Cosine branch in `search_layer0_bitset_fast`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn test_search_layer0_bitset_fast_cosine_uses_slab_vectors() {
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::Cosine,
        dim: 4,
        data_type: crate::api::DataType::Float,
        params: crate::api::IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(200),
            ..Default::default()
        },
    };
    let n = 300;
    let mut rng = crate::faiss::test_utils::seeded_rng(99);
    let vectors: Vec<f32> = (0..n * 4)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    assert!(index.layer0_slab.is_enabled_for(n));

    // Filter nodes 100-149
    let mut bitset = crate::bitset::Bitset::new(n);
    for i in 100..150 {
        bitset.set(i);
    }
    let bitset_view = bitset.as_view();
    let query = [0.7_f32, 0.1, -0.3, 0.5];
    let ef = 50;

    // Reference
    index.prepare_search_query_context(&query);
    let mut scratch_ref = SearchScratch::new();
    let reference = index.search_layer_idx_with_bitset_scratch(
        &query, 0, 0, ef, &bitset_view, &mut scratch_ref,
    );
    index.clear_search_query_context();

    // Fast path
    index.prepare_search_query_context(&query);
    let mut scratch_fast = SearchScratch::new();
    let fast = index.search_layer0_bitset_fast(
        &query, 0, ef, &bitset_view, &mut scratch_fast,
    );
    index.clear_search_query_context();

    let ref_ids: std::collections::HashSet<usize> = reference.iter().map(|r| r.0).collect();
    let fast_ids: std::collections::HashSet<usize> = fast.iter().map(|r| r.0).collect();
    assert_eq!(ref_ids, fast_ids, "cosine slab bitset results must match shared path");

    // No filtered nodes
    for (idx, _) in &fast {
        assert!(!(*idx >= 100 && *idx < 150), "filtered node {} in results", idx);
    }
}
```

- [ ] **Step 2: Run test to confirm it passes with current fallback**

Run: `cargo test test_search_layer0_bitset_fast_cosine_uses_slab_vectors -- --nocapture 2>&1 | tail -20`

Expected: PASS (Task 2 fallback to `distance_to_4_idxs_dispatch` works correctly).

- [ ] **Step 3: Add slab-native cosine batch-4**

In `search_layer0_bitset_fast`, replace the Cosine/IP `else` branch in the batch-4 block:

Replace:
```rust
                } else {
                    // Cosine/IP: use batch-4 dispatch (handles norm)
                    match Self::distance_to_4_idxs_dispatch(
                        self,
                        query,
                        batch_indices,
                        query_ptr,
                        base_ptr,
                    ) {
                        Some(d) => d,
                        None => {
                            // fallback: scalar one-by-one
                            [
                                self.distance(query, batch_indices[0]),
                                self.distance(query, batch_indices[1]),
                                self.distance(query, batch_indices[2]),
                                self.distance(query, batch_indices[3]),
                            ]
                        }
                    }
                };
```

With:
```rust
                } else {
                    // Cosine/IP: batch-4 IP from slab vector pointers
                    let v0 = self.layer0_slab.vector_slice_for(batch_indices[0], self.dim);
                    let v1 = self.layer0_slab.vector_slice_for(batch_indices[1], self.dim);
                    let v2 = self.layer0_slab.vector_slice_for(batch_indices[2], self.dim);
                    let v3 = self.layer0_slab.vector_slice_for(batch_indices[3], self.dim);
                    let mut ips = simd::ip_batch_4(query, v0, v1, v2, v3);
                    if self.metric_type == MetricType::Ip {
                        for ip in &mut ips {
                            *ip = -*ip;
                        }
                    } else {
                        // Cosine: stored vectors are pre-normalized, divide by query norm
                        let q_norm = HNSW_COSINE_QUERY_NORM_TLS.with(|c| c.get());
                        if q_norm > 0.0 {
                            for ip in &mut ips {
                                *ip = 1.0 - *ip / q_norm;
                            }
                        } else {
                            ips.fill(1.0);
                        }
                    }
                    ips
                };
```

This requires `layer0_slab.vector_slice_for()`. Check if it exists; if not, add it.

- [ ] **Step 4: Add `vector_slice_for` to `Layer0Slab` if missing**

Check if `Layer0Slab` has a method to get a `&[f32]` slice. If it only has `vector_ptr_for` (returns `*const f32`), add:

```rust
/// Return a slice of the vector data for the given node index.
fn vector_slice_for(&self, node_idx: usize, dim: usize) -> &[f32] {
    let base = node_idx * self.stride_words + self.vector_offset_words;
    let raw = &self.words[base..base + dim];
    // Safety: f32 and u32 have same size and alignment; stored as to_bits()
    unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, dim) }
}
```

Add this inside the `impl Layer0Slab` block (near `vector_ptr_for`).

- [ ] **Step 5: Run tests**

Run: `cargo test test_search_layer0_bitset_fast_cosine_uses_slab_vectors -- --nocapture 2>&1 | tail -20`

Expected: PASS

- [ ] **Step 6: Run full suite**

Run: `cargo test hnsw -- 2>&1 | tail -5`

Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): slab-native cosine batch-4 IP in bitset fast path

Uses layer0_slab vector slices directly for ip_batch_4 instead of reading
from self.vectors via index dispatch. Eliminates vector_start_offset bounds
checking and uses contiguous slab memory for better cache behavior.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: Local Benchmark Validation

Run the standalone HNSW benchmark to confirm the changes produce measurable QPS improvement on Cosine workload.

**Files:**
- No code changes. Benchmark only.

- [ ] **Step 1: Run local dev benchmark**

Run: `cargo run --example benchmark --release 2>&1 | grep -E "HNSW|hnsw|qps|QPS"`

Record the QPS numbers for HNSW 10K and 1M configurations.

- [ ] **Step 2: Compare with baseline**

Reference baselines from CLAUDE.md:
- HNSW 10K: 27,505 QPS (ef=50)
- HNSW 1M: 6,374 QPS (ef=50)

The standalone benchmark uses L2, not Cosine, so it validates that L2 slab path wasn't broken. For Cosine-specific validation, the Milvus VectorDBBench integration test is needed (Task 6).

- [ ] **Step 3: Write a quick standalone cosine QPS microbench**

Add a small test (not committed — local validation only):

```rust
#[test]
#[ignore] // Run manually: cargo test cosine_bitset_qps_micro -- --ignored --nocapture
fn cosine_bitset_qps_micro() {
    let dim = 128;
    let n = 100_000;
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::Cosine,
        dim,
        data_type: crate::api::DataType::Float,
        params: crate::api::IndexParams {
            m: Some(16),
            ef_construction: Some(200),
            ef_search: Some(128),
            ..Default::default()
        },
    };
    let mut rng = crate::faiss::test_utils::seeded_rng(42);
    let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();

    let bitset = crate::bitset::Bitset::new(n);
    let bitset_view = bitset.as_view();

    let n_queries = 1000;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let start = std::time::Instant::now();
    for q in &queries {
        let _ = index.search_single_with_bitset(q, 128, 10, &bitset_view);
    }
    let elapsed = start.elapsed();
    let qps = n_queries as f64 / elapsed.as_secs_f64();
    eprintln!("Cosine bitset search: {:.0} QPS over {} queries (100K, dim=128)", qps, n_queries);
}
```

- [ ] **Step 4: Run the microbench**

Run: `cargo test cosine_bitset_qps_micro -- --ignored --nocapture --release 2>&1 | tail -5`

Record QPS. This establishes a local before/after comparison baseline.

- [ ] **Step 5: Remove the microbench test (don't commit it)**

Delete the `cosine_bitset_qps_micro` test — it was for local validation only.

- [ ] **Step 6: Commit test cleanup if any test changes were made**

If tests were added in previous tasks that need minor fixes, commit them now. Otherwise skip.

---

### Task 6: Integration Validation (Milvus VectorDBBench)

Run the authority Milvus VectorDBBench 500K OpenAI comparison on x86 to validate the QPS improvement end-to-end. **This task requires remote x86 access and is optional if remote is unavailable.**

**Files:**
- No code changes.
- Rebuild `libknowhere_rs` on x86 with the new code.

- [ ] **Step 1: Push changes to x86**

Sync the changes to the x86 build machine:

```bash
rsync -avz --exclude target/ --exclude .git/ \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/
```

- [ ] **Step 2: Rebuild on x86**

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/knowhere-rs && \
  source "$HOME/.cargo/env" && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  cargo build --release --lib 2>&1 | tail -5'
```

Expected: Successful build.

- [ ] **Step 3: Restart Milvus standalone**

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src && \
  scripts/knowhere-rs-shim/start_standalone_remote.sh'
```

Wait for healthz.

- [ ] **Step 4: Run 500K OpenAI RS benchmark**

```bash
ssh hannsdb-x86 'cd /data/work/VectorDBBench && \
  python3 run_milvus_hnsw_500k_rs.py 2>&1 | tail -20'
```

- [ ] **Step 5: Collect and compare results**

```bash
ssh hannsdb-x86 'ls -t /data/work/VectorDBBench/vectordb_bench/results/Milvus/ | head -5'
```

Read the latest result JSON. Compare QPS/recall against:
- **Native baseline**: QPS=420.4, recall=0.9869
- **RS old baseline**: QPS=493.6, recall=0.9855
- **RS pre-fix**: QPS=282.7, recall=0.9705

**Target**: QPS ≥ 400 (close to native) with recall ≥ 0.97.
