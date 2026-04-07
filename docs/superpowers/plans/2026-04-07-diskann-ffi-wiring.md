# DiskANN FFI Wiring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Wire `CIndexType::DiskAnn` in `src/ffi.rs` to the existing `PQFlashIndex` implementation so Milvus can use RS DiskANN instead of falling back to brute-force.

**Architecture:** `IndexWrapper` already holds optional fields for every index type (flat, hnsw, scann, …). This plan adds `diskann: Option<PQFlashIndex>` to that struct, wires `new()` / `add()` / `train()` / `search()` / `search_with_bitset()` / `set_ef_search()` / `save()` / `load()`, and validates with a unit test. No new files. No ABI changes — `CIndexConfig.ef_construction` maps to `max_degree` and `ef_search` maps to `search_list_size` for DiskANN (same field reuse pattern as HNSW's ef_construction/ef_search).

**Tech Stack:** Rust, `src/ffi.rs`, `src/faiss/diskann_aisaq.rs::PQFlashIndex`

---

### Task 1: Add `diskann` field to `IndexWrapper` + fix all constructors

**Files:**
- Modify: `src/ffi.rs` — struct definition (line 297) and every `Some(Self { ... })` in `new()`

- [ ] **Step 1: Add field to `IndexWrapper` struct**

Find line 296–315 in `src/ffi.rs`:
```rust
/// 包装索引对象 - 支持 Flat, HNSW, ScaNN, HNSW-PRQ, IVF-RaBitQ, HNSW-SQ, HNSW-PQ, BinFlat, BinaryHnsw, IVF-SQ8, BinIvfFlat, SparseWand, SparseWandCC, MinHashLSH
struct IndexWrapper {
    flat: Option<MemIndex>,
    hnsw: Option<HnswIndex>,
    scann: Option<ScaNNIndex>,
    hnsw_prq: Option<crate::faiss::HnswPrqIndex>,
    hnsw_sq: Option<crate::faiss::HnswSqIndex>,
    hnsw_pq: Option<crate::faiss::HnswPqIndex>,
    ivf_pq: Option<crate::faiss::IvfPqIndex>,
    bin_flat: Option<crate::faiss::BinFlatIndex>,
    binary_hnsw: Option<crate::faiss::BinaryHnswIndex>,
    ivf_sq8: Option<crate::faiss::IvfSq8Index>,
    ivf_flat: Option<crate::faiss::IvfFlatIndex>,
    bin_ivf_flat: Option<crate::faiss::BinIvfFlatIndex>,
    sparse_inverted: Option<crate::faiss::SparseInvertedIndex>,
    sparse_wand: Option<crate::faiss::SparseWandIndex>,
    sparse_wand_cc: Option<crate::faiss::SparseWandIndexCC>,
    minhash_lsh: Option<crate::index::MinHashLSHIndex>,
    dim: usize,
}
```

Replace with (add `diskann` field before `dim`):
```rust
/// 包装索引对象 - 支持 Flat, HNSW, ScaNN, HNSW-PRQ, IVF-RaBitQ, HNSW-SQ, HNSW-PQ, BinFlat, BinaryHnsw, IVF-SQ8, BinIvfFlat, SparseWand, SparseWandCC, MinHashLSH, DiskANN
struct IndexWrapper {
    flat: Option<MemIndex>,
    hnsw: Option<HnswIndex>,
    scann: Option<ScaNNIndex>,
    hnsw_prq: Option<crate::faiss::HnswPrqIndex>,
    hnsw_sq: Option<crate::faiss::HnswSqIndex>,
    hnsw_pq: Option<crate::faiss::HnswPqIndex>,
    ivf_pq: Option<crate::faiss::IvfPqIndex>,
    bin_flat: Option<crate::faiss::BinFlatIndex>,
    binary_hnsw: Option<crate::faiss::BinaryHnswIndex>,
    ivf_sq8: Option<crate::faiss::IvfSq8Index>,
    ivf_flat: Option<crate::faiss::IvfFlatIndex>,
    bin_ivf_flat: Option<crate::faiss::BinIvfFlatIndex>,
    sparse_inverted: Option<crate::faiss::SparseInvertedIndex>,
    sparse_wand: Option<crate::faiss::SparseWandIndex>,
    sparse_wand_cc: Option<crate::faiss::SparseWandIndexCC>,
    minhash_lsh: Option<crate::index::MinHashLSHIndex>,
    diskann: Option<crate::faiss::diskann_aisaq::PQFlashIndex>,
    dim: usize,
}
```

- [ ] **Step 2: Add `diskann: None` to ALL existing `Some(Self { ... })` constructors**

Every existing arm in `fn new()` that returns `Some(Self { ..., minhash_lsh: None, dim })` must have `diskann: None` added before `dim`. There are ~16 such constructors. Use search + replace.

Run this to count how many need updating:
```bash
grep -n "minhash_lsh: None," src/ffi.rs | wc -l
```

For each one, add `diskann: None,` after `minhash_lsh: None,`:
```rust
            minhash_lsh: None,
            diskann: None,
            dim,
```

- [ ] **Step 3: Build to confirm all constructors are fixed**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build 2>&1 | grep "^error" | head -20
```

Expected: zero errors. If "missing field `diskann`" errors appear, fix those struct literals.

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "$(cat <<'EOF'
feat(ffi): add diskann field to IndexWrapper struct

Adds diskann: Option<PQFlashIndex> to IndexWrapper. All existing
constructors updated with diskann: None. Not yet wired in new().

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Wire `CIndexType::DiskAnn` in `fn new()`

**Files:**
- Modify: `src/ffi.rs` — lines 1075–1078 (the `DiskAnn` stub arm)

- [ ] **Step 1: Write a minimal unit test FIRST (TDD)**

Add this test to the `#[cfg(test)]` block in `src/ffi.rs` (near the bottom, around line 5100+):

```rust
#[test]
fn test_diskann_ffi_create() {
    // DiskANN index creation must succeed (was returning None before this fix)
    let config = CIndexConfig {
        index_type: CIndexType::DiskAnn,
        metric_type: CMetricType::L2,
        dim: 8,
        ef_construction: 16,  // reused as max_degree
        ef_search: 32,         // reused as search_list_size
        ..CIndexConfig::default()
    };
    let index = unsafe { knowhere_create_index(config) };
    assert!(
        !index.is_null(),
        "DiskANN index creation must return non-null"
    );
    unsafe { knowhere_free_index(index as *mut _) };
}
```

- [ ] **Step 2: Run test — verify it FAILS on current code**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo test test_diskann_ffi_create -- --nocapture 2>&1 | tail -5
```

Expected: `FAILED` — `assert!(!index.is_null())` because the stub returns `None`.

- [ ] **Step 3: Replace the stub with real DiskANN construction**

Find (lines 1075–1078):
```rust
            CIndexType::DiskAnn => {
                eprintln!("DiskANN not yet fully implemented via FFI");
                None
            }
```

Replace with:
```rust
            CIndexType::DiskAnn => {
                use crate::faiss::diskann_aisaq::{AisaqConfig, PQFlashIndex};
                let max_degree = if config.ef_construction > 0 {
                    config.ef_construction
                } else {
                    48
                };
                let search_list_size = if config.ef_search > 0 {
                    config.ef_search
                } else {
                    128
                };
                let aisaq_config = AisaqConfig {
                    max_degree,
                    search_list_size,
                    disk_pq_dims: 0, // in-memory mode, no disk PQ
                    ..AisaqConfig::default()
                };
                let diskann = PQFlashIndex::new(aisaq_config, metric, dim).ok()?;
                Some(Self {
                    flat: None,
                    hnsw: None,
                    scann: None,
                    hnsw_prq: None,
                    hnsw_sq: None,
                    hnsw_pq: None,
                    ivf_pq: None,
                    bin_flat: None,
                    binary_hnsw: None,
                    ivf_sq8: None,
                    ivf_flat: None,
                    bin_ivf_flat: None,
                    sparse_inverted: None,
                    sparse_wand: None,
                    sparse_wand_cc: None,
                    minhash_lsh: None,
                    diskann: Some(diskann),
                    dim,
                })
            }
```

- [ ] **Step 4: Build**

```bash
cargo build 2>&1 | grep "^error" | head -10
```

Expected: zero errors.

- [ ] **Step 5: Run the creation test — must pass now**

```bash
cargo test test_diskann_ffi_create -- --nocapture 2>&1 | tail -5
```

Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add src/ffi.rs
git commit -m "$(cat <<'EOF'
feat(ffi): wire CIndexType::DiskAnn to PQFlashIndex in IndexWrapper::new()

Replaces the eprintln!/None stub with real PQFlashIndex construction.
ef_construction -> max_degree, ef_search -> search_list_size.
disk_pq_dims=0 (no PQ, in-memory mode).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire `train()` and `add()`

**Files:**
- Modify: `src/ffi.rs` — `fn train()` (line 1220) and `fn add()` (line 1083)

- [ ] **Step 1: Add DiskANN arm to `fn train()`**

Find in `fn train()` (around line 1248):
```rust
        } else if self.sparse_inverted.is_some() || self.sparse_wand.is_some() {
            let _ = vectors;
            Ok(())
        } else {
            Err(CError::InvalidArg)
        }
```

Replace with:
```rust
        } else if self.sparse_inverted.is_some() || self.sparse_wand.is_some() {
            let _ = vectors;
            Ok(())
        } else if let Some(ref mut idx) = self.diskann {
            idx.train(vectors).map_err(|_| CError::Internal)
        } else {
            Err(CError::InvalidArg)
        }
```

- [ ] **Step 2: Add DiskANN arm to `fn add()`**

Find the last `else` in `fn add()` (search for `Ok(n_vectors)` followed by `} else {`). Add before the final `else`:

After the last existing `} else if let Some(ref mut idx) = self.minhash_lsh {` arm (or whatever is last), add:
```rust
        } else if let Some(ref mut idx) = self.diskann {
            idx.add_with_ids(vectors, ids).map_err(|_| CError::Internal)?;
            Ok(vectors.len() / self.dim)
```

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | grep "^error" | head -10
```

Expected: zero errors.

- [ ] **Step 4: Write and run correctness test (full create → train → add → search cycle)**

Add this test to the `#[cfg(test)]` block:

```rust
#[test]
fn test_diskann_ffi_add_and_search() {
    unsafe {
        let dim = 8usize;
        let n = 200usize;

        let config = CIndexConfig {
            index_type: CIndexType::DiskAnn,
            metric_type: CMetricType::L2,
            dim,
            ef_construction: 16,
            ef_search: 40,
            ..CIndexConfig::default()
        };
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        // Create simple vectors: vector i has component i/n at position i%dim
        let mut vectors: Vec<f32> = Vec::with_capacity(n * dim);
        for i in 0..n {
            for d in 0..dim {
                vectors.push(if d == i % dim { 1.0 } else { 0.0 });
            }
        }
        let ids: Vec<i64> = (0..n as i64).collect();

        // Train
        let err = knowhere_train_index(index as *mut _, vectors.as_ptr(), n, dim);
        assert_eq!(err, 0, "train must succeed");

        // Add
        let err = knowhere_add_index(index as *mut _, vectors.as_ptr(), ids.as_ptr(), n, dim);
        assert_eq!(err, 0, "add must succeed");

        // Search: query = unit vector at position 0 — nearest should be id=0
        let query: Vec<f32> = {
            let mut q = vec![0.0f32; dim];
            q[0] = 1.0;
            q
        };
        let result = knowhere_search(index, query.as_ptr(), 1, 3, dim);
        assert!(!result.is_null(), "search must return results");
        let result_ref = &*result;
        assert!(result_ref.num_results > 0, "must return at least 1 result");
        // The nearest result should be id=0 (exact match) or nearby
        let top_id = *result_ref.ids;
        assert!(top_id >= 0 && (top_id as usize) < n, "top id must be valid: got {}", top_id);

        knowhere_free_result(result);
        knowhere_free_index(index as *mut _);
    }
}
```

Run it:
```bash
cargo test test_diskann_ffi_add_and_search -- --nocapture 2>&1 | tail -8
```

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs
git commit -m "$(cat <<'EOF'
feat(ffi): wire DiskANN train() and add() in IndexWrapper

train() -> PQFlashIndex::train()
add() -> PQFlashIndex::add_with_ids(vectors, ids)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire `search()` in `IndexWrapper::search()` and `set_ef_search()`

**Files:**
- Modify: `src/ffi.rs` — `fn search()` (line 1256) and `fn set_ef_search()` (line 1370)

- [ ] **Step 1: Add DiskANN arm to `fn search()`**

Find the final `else` in `fn search()` (line ~1365):
```rust
        } else {
            Err(CError::InvalidArg)
        }
```

Add before it:
```rust
        } else if let Some(ref idx) = self.diskann {
            let start = std::time::Instant::now();
            let n_queries = query.len() / self.dim;
            let result = if n_queries <= 1 {
                idx.search(query, top_k).map_err(|_| CError::Internal)?
            } else {
                idx.search_batch(query, top_k).map_err(|_| CError::Internal)?
            };
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(result.ids, result.distances, elapsed_ms))
```

Note: `search_batch` is only available with the `parallel` feature. Check if `#[cfg(feature = "parallel")]` applies — if so, use `search` in a loop for the fallback. The simplest approach that always compiles:

```rust
        } else if let Some(ref idx) = self.diskann {
            let start = std::time::Instant::now();
            let result = idx.search_batch(query, top_k).map_err(|_| CError::Internal)?;
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(result.ids, result.distances, elapsed_ms))
```

If `search_batch` is feature-gated, use this fallback version instead:
```rust
        } else if let Some(ref idx) = self.diskann {
            let start = std::time::Instant::now();
            let n_queries = query.len() / self.dim;
            let top_k_inner = top_k;
            let mut all_ids = Vec::with_capacity(n_queries * top_k_inner);
            let mut all_dists = Vec::with_capacity(n_queries * top_k_inner);
            for q in query.chunks(self.dim) {
                let r = idx.search(q, top_k_inner).map_err(|_| CError::Internal)?;
                all_ids.extend_from_slice(&r.ids);
                all_dists.extend_from_slice(&r.distances);
            }
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
            Ok(ApiSearchResult::new(all_ids, all_dists, elapsed_ms))
```

Use whichever compiles. Prefer `search_batch` if available.

- [ ] **Step 2: Add DiskANN arm to `fn set_ef_search()`**

Find `fn set_ef_search()` (line 1370):
```rust
    fn set_ef_search(&mut self, ef_search: usize) -> Result<(), CError> {
        if let Some(ref mut idx) = self.hnsw {
            idx.set_ef_search(ef_search);
            Ok(())
        } else {
            Err(CError::InvalidArg)
        }
    }
```

Replace with:
```rust
    fn set_ef_search(&mut self, ef_search: usize) -> Result<(), CError> {
        if let Some(ref mut idx) = self.hnsw {
            idx.set_ef_search(ef_search);
            Ok(())
        } else if let Some(ref mut idx) = self.diskann {
            idx.set_search_list_size(ef_search);
            Ok(())
        } else {
            Err(CError::InvalidArg)
        }
    }
```

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | grep "^error" | head -10
```

Expected: zero errors. If `search_batch` is not found, fall back to the loop version.

- [ ] **Step 4: Run correctness test**

```bash
cargo test test_diskann_ffi_add_and_search -- --nocapture 2>&1 | tail -5
cargo test test_diskann_ffi_create -- --nocapture 2>&1 | tail -3
```

Expected: both `ok`.

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs
git commit -m "$(cat <<'EOF'
feat(ffi): wire DiskANN search() and set_ef_search() in IndexWrapper

search() -> PQFlashIndex::search_batch() (serial loop fallback if needed)
set_ef_search() -> PQFlashIndex::set_search_list_size()

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Wire `search_with_bitset` in the top-level FFI function

**Files:**
- Modify: `src/ffi.rs` — `knowhere_search_with_bitset()` (line 2507)

- [ ] **Step 1: Add DiskANN arm before the final `else` in `knowhere_search_with_bitset`**

Find (line ~2673):
```rust
        } else {
            // Do not silently drop bitset on unsupported index types.
            eprintln!("search_with_bitset not supported for this index type");
            std::ptr::null_mut()
        }
```

Add before it:
```rust
        } else if let Some(ref idx) = index.diskann {
            let bitset_view = crate::bitset::BitsetView::from_vec(
                bitset_words.to_vec(),
                bitset_wrapper.len,
            );
            let search_result = {
                let n_queries = query_slice.len() / index.dim;
                let top_k = req.top_k;
                let mut all_ids = Vec::with_capacity(n_queries * top_k);
                let mut all_dists = Vec::with_capacity(n_queries * top_k);
                let mut ok = true;
                for q in query_slice.chunks(index.dim) {
                    match idx.search_with_bitset(q, top_k, &bitset_view) {
                        Ok(r) => {
                            all_ids.extend_from_slice(&r.ids);
                            all_dists.extend_from_slice(&r.distances);
                        }
                        Err(_) => { ok = false; break; }
                    }
                }
                if ok { Ok((all_ids, all_dists)) } else { Err(()) }
            };
            match search_result {
                Ok((mut ids, mut distances)) => {
                    let num_results = ids.len();
                    let ids_ptr = ids.as_mut_ptr();
                    let distances_ptr = distances.as_mut_ptr();
                    std::mem::forget(ids);
                    std::mem::forget(distances);
                    let csr = CSearchResult {
                        ids: ids_ptr,
                        distances: distances_ptr,
                        num_results,
                        elapsed_ms: 0.0,
                    };
                    Box::into_raw(Box::new(csr))
                }
                Err(()) => std::ptr::null_mut(),
            }
```

Note: `search_with_bitset` takes a single query, so we loop over nq queries. If `search_batch_with_bitset` is available (parallel feature), prefer it.

- [ ] **Step 2: Build**

```bash
cargo build 2>&1 | grep "^error" | head -10
```

Expected: zero errors.

- [ ] **Step 3: Run all DiskANN FFI tests**

```bash
cargo test diskann_ffi 2>&1 | tail -8
```

Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "$(cat <<'EOF'
feat(ffi): wire DiskANN search_with_bitset in knowhere_search_with_bitset

Loop per-query search_with_bitset over bitset-filtered segments.
Applied before the final unsupported-type fallback.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Wire `save()` and `load()`

**Files:**
- Modify: `src/ffi.rs` — `fn save()` (line 2181) and `fn load()` (line 2216)

- [ ] **Step 1: Add DiskANN arm to `fn save()`**

Find the final `else` in `fn save()`:
```rust
        } else {
            Err(CError::InvalidArg)
        }
```

Add before it:
```rust
        } else if let Some(ref idx) = self.diskann {
            idx.save(path).map(|_| ()).map_err(|_| CError::Internal)
```

- [ ] **Step 2: Add DiskANN arm to `fn load()`**

Find the final `else` in `fn load()`:
```rust
        } else {
            Err(CError::InvalidArg)
        }
```

Add before it:
```rust
        } else if let Some(ref mut idx) = self.diskann {
            *idx = crate::faiss::diskann_aisaq::PQFlashIndex::load(path)
                .map_err(|_| CError::Internal)?;
            Ok(())
```

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | grep "^error" | head -10
```

Expected: zero errors.

- [ ] **Step 4: Write and run save/load roundtrip test**

Add to `#[cfg(test)]`:

```rust
#[test]
fn test_diskann_ffi_save_load_roundtrip() {
    use std::path::PathBuf;
    let tmp_dir = std::env::temp_dir().join("diskann_ffi_test");
    std::fs::create_dir_all(&tmp_dir).unwrap();
    let save_path = tmp_dir.to_str().unwrap();

    unsafe {
        let dim = 8usize;
        let n = 50usize;

        let config = CIndexConfig {
            index_type: CIndexType::DiskAnn,
            metric_type: CMetricType::L2,
            dim,
            ef_construction: 12,
            ef_search: 30,
            ..CIndexConfig::default()
        };
        let index = knowhere_create_index(config);
        assert!(!index.is_null());

        let vectors: Vec<f32> = (0..n * dim).map(|i| i as f32).collect();
        let ids: Vec<i64> = (0..n as i64).collect();

        let err = knowhere_train_index(index as *mut _, vectors.as_ptr(), n, dim);
        assert_eq!(err, 0);
        let err = knowhere_add_index(index as *mut _, vectors.as_ptr(), ids.as_ptr(), n, dim);
        assert_eq!(err, 0);

        // Save
        let path_cstr = std::ffi::CString::new(save_path).unwrap();
        let err = knowhere_save_index(index, path_cstr.as_ptr());
        assert_eq!(err, 0, "save must succeed");

        knowhere_free_index(index as *mut _);

        // Load into a fresh index
        let config2 = CIndexConfig {
            index_type: CIndexType::DiskAnn,
            metric_type: CMetricType::L2,
            dim,
            ef_construction: 12,
            ef_search: 30,
            ..CIndexConfig::default()
        };
        let loaded = knowhere_create_index(config2);
        assert!(!loaded.is_null());

        let err = knowhere_load_index(loaded as *mut _, path_cstr.as_ptr());
        assert_eq!(err, 0, "load must succeed");

        // Search on loaded index
        let query: Vec<f32> = vec![0.0f32; dim];
        let result = knowhere_search(loaded, query.as_ptr(), 1, 3, dim);
        assert!(!result.is_null(), "search on loaded index must succeed");
        let result_ref = &*result;
        assert!(result_ref.num_results > 0);

        knowhere_free_result(result);
        knowhere_free_index(loaded as *mut _);
    }

    // Cleanup
    let _ = std::fs::remove_dir_all(tmp_dir);
}
```

Run:
```bash
cargo test test_diskann_ffi_save_load_roundtrip -- --nocapture 2>&1 | tail -8
```

Expected: `ok`. If `knowhere_save_index` or `knowhere_load_index` have different signatures, adjust to match the actual FFI function names (search for `fn knowhere_save_index` and `fn knowhere_load_index` in ffi.rs).

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs
git commit -m "$(cat <<'EOF'
feat(ffi): wire DiskANN save() and load() in IndexWrapper

save() -> PQFlashIndex::save(path) (multi-file directory group)
load() -> PQFlashIndex::load(path) (replaces in-place)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Run full DiskANN FFI test suite + local benchmark sanity check

**Files:** None (verification only)

- [ ] **Step 1: Run all FFI tests**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo test diskann 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 2: Sanity check local benchmark (10K dev scale)**

```bash
cargo run --example benchmark --release 2>&1 | grep -i "diskann\|aisaq\|pqflash" | head -10
```

Confirm DiskANN benchmark still runs (standalone path, unaffected by FFI changes).

- [ ] **Step 3: Run full test suite to check no regressions**

```bash
cargo test 2>&1 | tail -5
```

Expected: no failures.

---

### Task 8: Sync to x86, build release, restart Milvus, re-run DiskANN benchmark

**Files:** None (remote execution)

- [ ] **Step 1: Sync and build on x86**

```bash
rsync -av --delete \
  --exclude='.git' \
  --exclude='target' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/

ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/knowhere-rs && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  ~/.cargo/bin/cargo build --release 2>&1 | tail -3"
```

Expected: `Finished release [optimized] target(s) in ...`

- [ ] **Step 2: Restart Milvus**

```bash
ssh hannsdb-x86 "pkill -f 'milvus run standalone' || true"
sleep 5
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/milvus-src && \
  nohup bin/milvus run standalone > /tmp/milvus_diskann_r1.log 2>&1 &"
sleep 30
```

- [ ] **Step 3: Drop existing DiskANN collection and rebuild**

```bash
ssh hannsdb-x86 "cd /data/work/VectorDBBench && .venv/bin/python3 -c \"
from pymilvus import connections, Collection, utility
connections.connect(host='127.0.0.1', port='19530')
if utility.has_collection('diskann_rs_bench'):
    utility.drop_collection('diskann_rs_bench')
    print('Dropped diskann_rs_bench')
\" 2>&1"
```

Then re-run the setup script (same synthetic 1M × 768-dim normalized vectors, DiskANN build):
```bash
ssh hannsdb-x86 "nohup /data/work/VectorDBBench/.venv/bin/python3 /tmp/diskann_setup.py > /tmp/diskann_setup_r1.log 2>&1 &"
echo "Build started; wait ~10-40 min"
```

Wait for build:
```bash
for i in $(seq 1 30); do
  sleep 120
  ssh hannsdb-x86 "grep 'RESULT_SETUP:\|Error\|Traceback' /tmp/diskann_setup_r1.log 2>/dev/null | head -3"
  ssh hannsdb-x86 "grep -q 'RESULT_SETUP:' /tmp/diskann_setup_r1.log" && break
done
ssh hannsdb-x86 "tail -5 /tmp/diskann_setup_r1.log"
```

**Key diagnostic:** build time should now be 10+ minutes (real graph construction) instead of 16s (stub). If still fast, check Milvus log for errors.

- [ ] **Step 4: Load collection and run QPS benchmark**

```bash
ssh hannsdb-x86 "nohup /data/work/VectorDBBench/.venv/bin/python3 /tmp/diskann_bench.py > /tmp/diskann_bench_r1.log 2>&1 &"
echo "QPS benchmark started..."
sleep 350
ssh hannsdb-x86 "cat /tmp/diskann_bench_r1.log"
```

- [ ] **Step 5: Check latency (expect < 20ms/query for real graph search vs 414ms for brute-force)**

```bash
ssh hannsdb-x86 "cd /data/work/VectorDBBench && .venv/bin/python3 -c \"
from pymilvus import connections, Collection
import time, numpy as np
connections.connect(host='127.0.0.1', port='19530')
col = Collection('diskann_rs_bench')
q = np.random.randn(1, 768).astype('float32').tolist()
params = {'metric_type': 'IP', 'params': {'search_list': 100}}
for _ in range(5): col.search(q, 'vector', params, limit=10, output_fields=[])
times = []
for _ in range(20):
    t0 = time.time()
    col.search(q, 'vector', params, limit=10, output_fields=[])
    times.append((time.time()-t0)*1000)
print(f'p50={sorted(times)[10]:.1f}ms p99={sorted(times)[18]:.1f}ms')
\" 2>&1"
```

Expected: p50 < 50ms (graph search), not 414ms (brute-force).

- [ ] **Step 6: Record results and update benchmark file**

Update `benchmark_results/diskann_milvus_rs_2026-04-07.md` with real R1 numbers:
- Build time (real graph build)
- p50/p99 latency
- c=1/c=20/c=80 QPS
- Recall (re-run `/tmp/diskann_recall.py`)

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
git add benchmark_results/diskann_milvus_rs_2026-04-07.md
git commit -m "$(cat <<'EOF'
bench(diskann): RS DiskANN Milvus R1 — real graph search results

Build=Xs latency_p50=Xms c=20=X c=80=X QPS recall=X
(FFI now wired to PQFlashIndex, not brute-force fallback)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
git push origin main
```

---

## Expected Outcome

After this plan:
- `CIndexType::DiskAnn` creates a real Vamana-graph `PQFlashIndex`
- Build time in Milvus: 10–40 min for 1M × 768-dim (real graph, vs 16s stub)
- Search latency p50: 5–50ms (beam search, vs 414ms brute-force)
- QPS c=80: target >200 QPS (vs 12 QPS from brute-force)
- Recall: depends on `search_list_size`; with 100 should be ≥ 0.9

If build still returns fast (< 60s) or latency is still ~400ms, the FFI call is not reaching the RS code — check Milvus log for `knowhere_create_index` failures and confirm the `.so` was updated.
