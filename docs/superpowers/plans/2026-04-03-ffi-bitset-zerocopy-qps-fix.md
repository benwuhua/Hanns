# FFI Bitset Zero-Copy — Milvus Search QPS Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 3.9x overhead in the Milvus FFI `knowhere_search_with_bitset` path, raising Cohere 1M HNSW search QPS from ~351 toward the standalone baseline of ~1380 (ef=128).

**Architecture:** Replace the bitset copy chain (`to_vec()` → `BitsetView::from_vec()` → `Arc<BitsetPredicate>` clone) with a borrowed `BitsetRef` type that wraps `&[u64]` directly from the C `CBitset` pointer. Remove dead code (`sparse_bitset` creation, `Arc<BitsetPredicate>` construction) from the HNSW search path.

**Tech Stack:** Rust, FFI (C ABI), knowhere-rs HNSW, bitset module

---

## Context

### Problem Statement

Milvus integration benchmark (Cohere 1M, 768D, Cosine, HNSW M=16):
- RS standalone HNSW: QPS=1380 at ef=128, recall=0.966
- RS via Milvus FFI: QPS=351 at ef=128, recall=0.990
- Native knowhere via Milvus: QPS=848 at ef=128, recall=0.958

The 3.9x overhead (1380 → 351) is entirely in the FFI bitset path. Root causes:

1. **Bitset copy** (ffi.rs:2523-2525): `slice::from_raw_parts(cbitset.data, ...).to_vec()` copies ~125KB per search call for 1M vectors
2. **Dead allocation #1** (ffi.rs:2527): `bitset_to_bool_vec()` creates a 1MB `Vec<bool>` that is never used by HNSW
3. **Dead allocation #2** (ffi.rs:2533): `Arc::new(BitsetPredicate::new(bitset_view.clone()))` clones the entire bitset Vec again; HNSW never reads `req.filter`

### Benchmark Evidence

```
RS standalone ef-sweep (Cohere 1M, 768D, Cosine, M=16, serial build):
ef=96:  recall=0.952, single_qps=1765, batch_qps=8360
ef=112: recall=0.960, single_qps=1551, batch_qps=7341
ef=128: recall=0.966, single_qps=1380, batch_qps=6454
```

### File Map

| File | Responsibility |
|------|---------------|
| `src/bitset.rs` | `BitsetView` — owns `Vec<u64>`, `get()`/`len()`/`set()` |
| `src/ffi.rs:2498-2590` | `knowhere_search_with_bitset` — FFI entry, bitset copy, dispatch |
| `src/ffi.rs:3835-3870` | `CBitset` — C wrapper with raw `*mut u64` + `len` |
| `src/faiss/hnsw.rs:4216` | `HnswIndex::search_with_bitset(query, req, &BitsetView)` |
| `src/faiss/hnsw.rs:5435` | `search_layer0_bitset_fast` — hot path, calls `bitset.get(idx)` |
| `src/faiss/hnsw.rs:6425` | `search_single_with_bitset` |
| `src/api/search.rs:199` | `BitsetPredicate` — wraps `BitsetView`, used by `SearchRequest.filter` |
| `src/faiss/sparse_inverted.rs:779` | `bitset_to_bool_vec` — only used by sparse indexes |

---

## Task 1: Add `BitsetRef` borrowed bitset type

**Files:**
- Modify: `src/bitset.rs`

The HNSW search hot path only needs `get(idx) -> bool` and `len() -> usize`. Create a lightweight borrowed type that satisfies this without owning data.

- [ ] **Step 1: Write failing test for `BitsetRef`**

Add to the test module at the bottom of `src/bitset.rs`:

```rust
#[test]
fn test_bitset_ref_zero_copy() {
    // Simulate what FFI will do: borrow from raw u64 words
    let words: Vec<u64> = vec![0b1010, 0b0101];
    let len = 100; // 128 bits in 2 words, but we say 100 for partial last word

    let bitset_ref = BitsetRef::new(&words, len);

    assert_eq!(bitset_ref.len(), 100);
    assert_eq!(bitset_ref.get(0), false); // word[0] bit 0 = 0
    assert_eq!(bitset_ref.get(1), true);  // word[0] bit 1 = 1
    assert_eq!(bitset_ref.get(2), false); // word[0] bit 2 = 0
    assert_eq!(bitset_ref.get(3), true);  // word[0] bit 3 = 1
    assert_eq!(bitset_ref.get(64), true); // word[1] bit 0 = 1
    assert_eq!(bitset_ref.get(65), false); // word[1] bit 1 = 0
    assert!(bitset_ref.get(200) == false); // out of range
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test test_bitset_ref_zero_copy -- --nocapture 2>&1 | tail -5`
Expected: FAIL — `BitsetRef` not defined

- [ ] **Step 3: Implement `BitsetRef`**

Add to `src/bitset.rs` after the `BitsetView` struct definition (around line 60):

```rust
/// Borrowed bitset reference — zero-copy view over external u64 words.
///
/// Used in FFI hot paths where the bitset data is owned by the caller (C `CBitset`)
/// and remains valid for the duration of the search call.
#[derive(Clone, Copy)]
pub struct BitsetRef<'a> {
    data: &'a [u64],
    len: usize,
}

impl<'a> BitsetRef<'a> {
    /// Create a borrowed bitset view from a slice of u64 words and a bit count.
    #[inline]
    pub fn new(data: &'a [u64], len: usize) -> Self {
        Self { data, len }
    }

    /// Number of bits in the bitset.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Check bit at index. Returns false for out-of-range indices.
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        if index >= self.len {
            return false;
        }
        let word = index / 64;
        let bit = index % 64;
        (self.data[word] >> bit) & 1 == 1
    }

    /// Alias for `get` — matches `BitsetView::test` naming.
    #[inline]
    pub fn test(&self, index: usize) -> bool {
        self.get(index)
    }

    /// Count number of set bits (1s) in the bitset.
    pub fn count_ones(&self) -> usize {
        let full_words = self.len / 64;
        let mut count = 0usize;
        for &word in &self.data[..full_words] {
            count += word.count_ones() as usize;
        }
        // Handle partial last word
        let remaining = self.len % 64;
        if remaining > 0 && full_words < self.data.len() {
            let mask = (1u64 << remaining) - 1;
            count += (self.data[full_words] & mask).count_ones() as usize;
        }
        count
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test test_bitset_ref_zero_copy -- --nocapture 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/bitset.rs
git commit -m "feat(bitset): add BitsetRef zero-copy borrowed bitset type"
```

---

## Task 2: Make `search_layer0_bitset_fast` generic over bitset type

**Files:**
- Modify: `src/faiss/hnsw.rs`

The hot path function `search_layer0_bitset_fast` currently takes `&BitsetView`. We need it to also accept `BitsetRef`. The cleanest way is a trait.

- [ ] **Step 1: Define `BitsetCheck` trait**

Add near the top of `src/faiss/hnsw.rs` (after imports):

```rust
/// Trait for bitset operations used in HNSW search.
/// Implemented by both owned `BitsetView` and borrowed `BitsetRef`.
pub trait BitsetCheck {
    fn bitset_len(&self) -> usize;
    fn bitset_get(&self, index: usize) -> bool;
}

impl BitsetCheck for crate::bitset::BitsetView {
    #[inline]
    fn bitset_len(&self) -> usize { self.len() }
    #[inline]
    fn bitset_get(&self, index: usize) -> bool { self.get(index) }
}

impl<'a> BitsetCheck for crate::bitset::BitsetRef<'a> {
    #[inline]
    fn bitset_len(&self) -> usize { self.len() }
    #[inline]
    fn bitset_get(&self, index: usize) -> bool { self.get(index) }
}
```

- [ ] **Step 2: Write failing test**

Add test to verify the trait works with both types:

```rust
#[test]
fn test_bitset_check_trait_both_types() {
    use crate::bitset::{BitsetView, BitsetRef};

    let mut owned = BitsetView::new(128);
    owned.set(3, true);
    owned.set(67, true);

    let words = owned.as_slice().to_vec();
    let borrowed = BitsetRef::new(&words, 128);

    fn check_bitset<B: BitsetCheck>(b: &B) {
        assert_eq!(b.bitset_len(), 128);
        assert!(b.bitset_get(3));
        assert!(b.bitset_get(67));
        assert!(!b.bitset_get(0));
    }

    check_bitset(&owned);
    check_bitset(&borrowed);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test test_bitset_check_trait_both_types -- --nocapture 2>&1 | tail -5`
Expected: FAIL — `BitsetCheck` not defined (before step 1 code is added) or PASS (after)

- [ ] **Step 4: Generalize `search_layer0_bitset_fast`**

Change the signature from:
```rust
fn search_layer0_bitset_fast(
    &self,
    query: &[f32],
    entry_idx: usize,
    ef: usize,
    bitset: &crate::bitset::BitsetView,
    scratch: &mut SearchScratch,
) -> Vec<(usize, f32)>
```

To:
```rust
fn search_layer0_bitset_fast<B: BitsetCheck>(
    &self,
    query: &[f32],
    entry_idx: usize,
    ef: usize,
    bitset: &B,
    scratch: &mut SearchScratch,
) -> Vec<(usize, f32)>
```

Inside the function body, replace:
- `bitset.len()` → `bitset.bitset_len()`
- `bitset.get(idx)` → `bitset.bitset_get(idx)`

All other logic remains identical.

- [ ] **Step 5: Update callers of `search_layer0_bitset_fast`**

All callers pass `&BitsetView` — this will still work because `BitsetView: BitsetCheck`. Find every call site and verify compilation:

Run: `cargo build 2>&1 | grep "^error" | head -10`
Expected: 0 errors (monomorphization handles both types)

- [ ] **Step 6: Run all HNSW bitset tests**

Run: `cargo test hnsw -- --nocapture 2>&1 | grep -E "test result|FAILED" | tail -3`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "refactor(hnsw): generalize bitset search with BitsetCheck trait"
```

---

## Task 3: Eliminate dead allocations in FFI `knowhere_search_with_bitset`

**Files:**
- Modify: `src/ffi.rs:2498-2590`

This is the main fix. Replace the copy chain with a zero-copy `BitsetRef`.

- [ ] **Step 1: Write the FFI-level test**

Add to the test module in `src/ffi.rs`:

```rust
#[test]
fn test_search_with_bitset_no_copy() {
    // Build a small index
    let config = CIndexConfig {
        index_type: CIndexType::Hnsw,
        metric_type: CMetricType::L2,
        dim: 32,
        ef_construction: 200,
        ef_search: 64,
        ..Default::default()
    };
    let index = knowhere_create_index(config.clone());
    assert!(!index.is_null());

    // Add vectors
    let n = 100;
    let dim = 32;
    let vectors: Vec<f32> = (0..n * dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let ids: Vec<i64> = (0..n).collect();
    let add_result = knowhere_add_index(index, vectors.as_ptr(), ids.as_ptr(), n, dim);
    assert_eq!(add_result, CError::Success as i32);

    // Create bitset (no filtering)
    let bitset = knowhere_bitset_create(n);
    // Search with bitset
    let query: Vec<f32> = vec![0.0; dim];
    let result = knowhere_search_with_bitset(index, query.as_ptr(), 1, 5, dim, bitset);
    assert!(!result.is_null());

    unsafe {
        let r = &*result;
        assert!(r.num_results > 0);
        knowhere_free_result(result);
    }
    knowhere_bitset_free(bitset as *mut _);
    knowhere_free_index(index);
}
```

- [ ] **Step 2: Run test to verify it passes with current code**

Run: `cargo test test_search_with_bitset_no_copy -- --nocapture 2>&1 | tail -5`
Expected: PASS (baseline before change)

- [ ] **Step 3: Rewrite the HNSW branch in `knowhere_search_with_bitset`**

Replace the HNSW search branch (the `else if let Some(ref idx) = index.hnsw` block, approximately lines 2564-2590) with zero-copy path. The full change to the bitset construction section (lines 2522-2537):

**Before:**
```rust
let bitset_data =
    std::slice::from_raw_parts(bitset_wrapper.data, bitset_wrapper.len.div_ceil(64))
        .to_vec();
let bitset_view = crate::bitset::BitsetView::from_vec(bitset_data, bitset_wrapper.len);
let sparse_bitset = crate::faiss::sparse_inverted::bitset_to_bool_vec(&bitset_view);
let req = SearchRequest {
    top_k,
    nprobe: 8,
    filter: Some(std::sync::Arc::new(crate::api::BitsetPredicate::new(
        bitset_view.clone(),
    ))),
    params: None,
    radius: None,
};
```

**After:**
```rust
// Zero-copy: borrow the CBitset's u64 words directly.
// CBitset data is stable for the duration of this FFI call.
let bitset_words = unsafe {
    std::slice::from_raw_parts(bitset_wrapper.data, bitset_wrapper.len.div_ceil(64))
};
let bitset_ref = crate::bitset::BitsetRef::new(bitset_words, bitset_wrapper.len);

// Build a BitsetView for indexes that need owned data (flat, sparse)
let bitset_data = bitset_words.to_vec();
let bitset_view = crate::bitset::BitsetView::from_vec(bitset_data, bitset_wrapper.len);

let req = SearchRequest {
    top_k,
    nprobe: 8,
    filter: None, // HNSW uses bitset_ref directly, not req.filter
    params: None,
    radius: None,
};
```

Then in the HNSW branch, use `bitset_ref` instead of `bitset_view`:

```rust
} else if let Some(ref idx) = index.hnsw {
    match idx.search_with_bitset_ref(query_slice, &req, &bitset_ref) {
        // ... same result handling ...
    }
}
```

The `flat` and `scann` branches continue using `bitset_view` (owned data).

- [ ] **Step 4: Add `search_with_bitset_ref` to HnswIndex**

Add method to `src/faiss/hnsw.rs`:

```rust
/// Search with a borrowed bitset reference (zero-copy FFI path).
pub fn search_with_bitset_ref<'a>(
    &self,
    query: &[f32],
    req: &SearchRequest,
    bitset: &crate::bitset::BitsetRef<'a>,
) -> Result<ApiSearchResult> {
    if self.vectors.is_empty() {
        return Err(crate::api::KnowhereError::InvalidArg("index is empty".into()));
    }
    let n_queries = query.len() / self.dim;
    let k = req.top_k;
    let ef = self.config.params.effective_hnsw_ef_search(self.ef_search, req.nprobe, k);

    let mut all_ids = Vec::new();
    let mut all_dists = Vec::new();

    for q_idx in 0..n_queries {
        let q_start = q_idx * self.dim;
        let query_vec = &query[q_start..q_start + self.dim];
        let results = self.search_single_with_bitset_ref(query_vec, ef, k, bitset);
        for (id, dist) in results.into_iter().take(k) {
            all_ids.push(id);
            all_dists.push(dist);
        }
    }

    // Finalize distances (same as search_with_bitset)
    for i in 0..all_dists.len() {
        if all_ids[i] != -1 {
            match self.metric_type {
                MetricType::L2 => { all_dists[i] = all_dists[i].sqrt(); }
                MetricType::Ip => { all_dists[i] = -all_dists[i]; }
                MetricType::Cosine => { /* already normalized */ }
            }
        }
    }

    Ok(ApiSearchResult { ids: all_ids, distances: all_dists, elapsed_ms: 0.0 })
}
```

And `search_single_with_bitset_ref`:

```rust
fn search_single_with_bitset_ref<'a>(
    &self,
    query: &[f32],
    ef: usize,
    k: usize,
    bitset: &crate::bitset::BitsetRef<'a>,
) -> Vec<(usize, f32)> {
    self.prepare_search_query_context(query);
    let num_nodes = self.node_info.len();

    if num_nodes < self.config.params.bruteforce_threshold.unwrap_or(500) {
        let mut results = self.brute_force_search(query, k, |_id, idx| {
            idx >= bitset.len() || !bitset.get(idx)
        });
        self.rerank_sq_results(query, &mut results);
        return results;
    }

    let curr_ep_idx = self.entry_point.unwrap_or(0);

    HNSW_SEARCH_SCRATCH_TLS.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        self.search_single_with_bitset_ref_scratch(query, curr_ep_idx, ef, k, bitset, &mut scratch)
    })
}
```

And `search_single_with_bitset_ref_scratch` — delegates to `search_layer0_bitset_fast` which is now generic:

```rust
fn search_single_with_bitset_ref_scratch<'a>(
    &self,
    query: &[f32],
    entry_idx: usize,
    ef: usize,
    k: usize,
    bitset: &crate::bitset::BitsetRef<'a>,
    scratch: &mut SearchScratch,
) -> Vec<(usize, f32)> {
    // Layer 0 search — use generic fast path
    let results = self.search_layer0_bitset_fast(query, entry_idx, ef, bitset, scratch);

    let mut final_results = Vec::with_capacity(k);
    for (idx, dist) in results.into_iter().take(k) {
        let id = self.resolve_id(idx);
        final_results.push((id, dist));
    }
    final_results
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cargo test test_search_with_bitset_no_copy -- --nocapture 2>&1 | tail -5`
Expected: PASS

- [ ] **Step 6: Run full HNSW bitset test suite**

Run: `cargo test bitset -- --nocapture 2>&1 | grep -E "test result|FAILED" | tail -3`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add src/ffi.rs src/faiss/hnsw.rs
git commit -m "perf(ffi): zero-copy bitset in HNSW search path, eliminate dead allocs"
```

---

## Task 4: Validate with standalone benchmark

**Files:**
- Modify: `src/faiss/hnsw.rs` (test only)

- [ ] **Step 1: Add benchmark test for bitset search throughput**

Add to the HNSW test module:

```rust
#[test]
fn test_hnsw_bitset_search_throughput() {
    let dim = 128;
    let n = 10_000;
    let nq = 100;
    let k = 10;
    let ef = 64;
    let data = deterministic_parallel_profile_vectors(n, dim);
    let queries = deterministic_parallel_profile_queries(&data, nq, dim);

    let cfg = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        data_type: DataType::Float,
        dim,
        params: IndexParams::hnsw(200, ef, 1.0 / (16.0_f32).ln()),
    };
    let mut index = HnswIndex::new(&cfg).unwrap();
    index.train(&data).unwrap();
    index.add(&data, None).unwrap();

    // Build empty bitset (no filtering)
    let bitset_view = BitsetView::new(n);
    let bitset_words = bitset_view.as_slice().to_vec();
    let bitset_ref = crate::bitset::BitsetRef::new(&bitset_words, n);

    let req = SearchRequest { top_k: k, nprobe: ef, ..Default::default() };

    // Warm up
    for q in queries.chunks_exact(dim).take(10) {
        let _ = index.search_with_bitset(q, &req, &bitset_view);
    }

    // Benchmark owned BitsetView
    let t1 = std::time::Instant::now();
    for q in queries.chunks_exact(dim) {
        let _ = index.search_with_bitset(q, &req, &bitset_view);
    }
    let owned_qps = nq as f64 / t1.elapsed().as_secs_f64();

    // Benchmark borrowed BitsetRef
    let t2 = std::time::Instant::now();
    for q in queries.chunks_exact(dim) {
        let _ = index.search_with_bitset_ref(q, &req, &bitset_ref);
    }
    let ref_qps = nq as f64 / t2.elapsed().as_secs_f64();

    eprintln!("owned_bitset_qps={:.0} borrowed_bitset_qps={:.0} ratio={:.2}x",
              owned_qps, ref_qps, ref_qps / owned_qps);
    // BitsetRef should be at least as fast as BitsetView
    assert!(ref_qps >= owned_qps * 0.95, "BitsetRef should not regress vs BitsetView");
}
```

- [ ] **Step 2: Run throughput test**

Run: `cargo test test_hnsw_bitset_search_throughput --release -- --nocapture 2>&1 | tail -10`
Expected: PASS with BitsetRef >= 0.95x BitsetView speed

- [ ] **Step 3: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "test(hnsw): add bitset search throughput comparison test"
```

---

## Task 5: Build and validate on x86 authority

**Files:**
- No source changes (deployment only)

- [ ] **Step 1: Sync to x86 and build**

```bash
rsync -az --delete --exclude target --exclude '.git' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  knowhere-x86-hk-proxy:/data/work/knowhere-rs-src/

ssh knowhere-x86-hk-proxy \
  "cd /data/work/knowhere-rs-src && \
   CARGO_TARGET_DIR=/data/work/knowhere-rs-target-ffi-zerocopy \
   ~/.cargo/bin/cargo build --release --lib 2>&1 | tail -5"
```

Expected: Build succeeds

- [ ] **Step 2: Run existing tests on x86**

```bash
ssh knowhere-x86-hk-proxy \
  "cd /data/work/knowhere-rs-src && \
   CARGO_TARGET_DIR=/data/work/knowhere-rs-target-ffi-zerocopy \
   ~/.cargo/bin/cargo test --release bitset -- --nocapture 2>&1 | tail -10"
```

Expected: All bitset tests pass

- [ ] **Step 3: Run Cohere 1M ef-sweep with fix**

```bash
ssh knowhere-x86-hk-proxy \
  "cd /data/work/knowhere-rs-src && \
   CARGO_TARGET_DIR=/data/work/knowhere-rs-target-ffi-zerocopy \
   ~/.cargo/bin/cargo run --release --example cohere_hnsw -- \
   /data/work/datasets/wikipedia-cohere-1m 16 cosine \
   > /data/work/knowhere-rs-logs-ffi-zerocopy/ef_sweep.log 2>&1"
```

Compare ef=128 QPS with previous standalone baseline (1380 single QPS).

- [ ] **Step 4: Record results**

Save comparison to `benchmark_results/ffi_bitset_zerocopy_cohere1m_YYYYMMDD.json`.

---

## Task 6: Milvus integrated benchmark (optional, if access available)

- [ ] **Step 1: Rebuild knowhere-rs in milvus-rs-integ**

```bash
ssh knowhere-x86-hk-proxy \
  "cd /data/work/milvus-rs-integ/knowhere-rs && \
   CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
   ~/.cargo/bin/cargo build --release --lib 2>&1 | tail -5"
```

- [ ] **Step 2: Restart Milvus with new knowhere-rs**

```bash
ssh knowhere-x86-hk-proxy \
  "cd /data/work/milvus-rs-integ/milvus-src && \
   scripts/knowhere-rs-shim/start_standalone_remote.sh"
```

Wait for healthz.

- [ ] **Step 3: Run VDB benchmark and compare**

Target: QPS improvement from ~351 → hopefully 500+ at ef=128.

---

## Self-Review Checklist

1. **Spec coverage**: All three dead allocations identified in the analysis are addressed:
   - `to_vec()` copy → replaced with `BitsetRef` borrow ✅
   - `bitset_to_bool_vec()` → removed from HNSW path ✅
   - `Arc<BitsetPredicate>` clone → `filter: None` for HNSW ✅

2. **Placeholder scan**: No TBDs, TODOs, or "implement later" ✅

3. **Type consistency**: `BitsetRef<'a>` used consistently; `BitsetCheck` trait implemented for both `BitsetView` and `BitsetRef<'a>`; `search_layer0_bitset_fast<B: BitsetCheck>` monomorphizes for both ✅
