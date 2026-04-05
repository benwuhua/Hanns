# Milvus HNSW Performance Optimization Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the Load (1.77x) and Optimize (1.65x) gaps vs native Milvus knowhere by fixing two root-cause bugs in `src/faiss/hnsw.rs`: per-f32 serialization (134M syscalls) and per-node heap allocation in serial add (1M allocs).

**Architecture:** Two independent changes in `src/faiss/hnsw.rs`. Fix 1 replaces per-element `write_all`/`read_exact` loops in `write_to` and `read_from` with 256KB-chunked byte-buffer I/O. Fix 2 replaces `to_vec()` per node in `add()` and `add_profiled()` with a single pre-allocated scratch buffer. A third change adds optional search tracing to `src/ffi.rs` for QPS profiling in Round 2. No data format changes; all fixes are backward-compatible with existing serialized indexes.

**Tech Stack:** Rust, `std::io::{Read, Write}`, no new dependencies.

---

## File Map

| File | Change |
|------|--------|
| `src/faiss/hnsw.rs` | Fix `write_to` (line ~6903), `read_from` (line ~7435, ~7469), `add` (line ~1982), `add_profiled` (line ~2063) |
| `src/ffi.rs` | Add optional search tracing in `knowhere_search` (line ~2395) |
| `tests/test_hnsw_bulk_io.rs` | New test file: round-trip correctness for serialization fix |

---

## Task 1: Write failing tests for bulk serialize/deserialize

**Files:**
- Create: `tests/test_hnsw_bulk_io.rs`

- [ ] **Step 1: Create the test file**

```rust
// tests/test_hnsw_bulk_io.rs
use knowhere_rs::api::{DataType, IndexConfig, IndexParams, SearchRequest};
use knowhere_rs::faiss::HnswIndex;
use knowhere_rs::{IndexType, MetricType};

const DIM: usize = 32;

fn make_vectors(n: usize) -> Vec<f32> {
    // Deterministic: v[i][j] = sin(i * dim + j) * 0.001
    (0..n * DIM)
        .map(|k| (k as f32 * 0.001).sin())
        .collect()
}

fn build_test_index(n: usize, metric: MetricType) -> HnswIndex {
    let vectors = make_vectors(n);
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: metric,
        dim: DIM,
        data_type: DataType::Float,
        params: IndexParams {
            m: Some(16),
            ef_construction: Some(64),
            ef_search: Some(64),
            ..Default::default()
        },
    };
    let mut index = HnswIndex::new(&config).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, None).unwrap();
    index
}

fn search_top_k(index: &HnswIndex, queries: &[f32], k: usize) -> Vec<Vec<i64>> {
    let n_queries = queries.len() / DIM;
    let req = SearchRequest {
        top_k: k,
        nprobe: 64,
        filter: None,
        params: None,
        radius: None,
    };
    let result = index.search(queries, &req).unwrap();
    (0..n_queries)
        .map(|q| result.ids[q * k..(q + 1) * k].to_vec())
        .collect()
}

/// Round-trip serialize → deserialize must return identical search results (L2).
#[test]
fn test_serialize_roundtrip_l2_search() {
    let index = build_test_index(500, MetricType::L2);
    let queries = make_vectors(10); // first 10 vectors as queries
    let k = 5;

    let before = search_top_k(&index, &queries, k);

    let bytes = index.serialize_to_bytes().unwrap();
    let restored = HnswIndex::deserialize_from_bytes(&bytes).unwrap();

    let after = search_top_k(&restored, &queries, k);

    assert_eq!(before, after,
        "search results changed after L2 serialize/deserialize roundtrip");
}

/// Round-trip for Cosine must also return identical results (normalization not double-applied).
#[test]
fn test_serialize_roundtrip_cosine_search() {
    let index = build_test_index(500, MetricType::Cosine);
    let queries = make_vectors(10);
    let k = 5;

    let before = search_top_k(&index, &queries, k);

    let bytes = index.serialize_to_bytes().unwrap();
    let restored = HnswIndex::deserialize_from_bytes(&bytes).unwrap();

    let after = search_top_k(&restored, &queries, k);

    assert_eq!(before, after,
        "search results changed after Cosine serialize/deserialize roundtrip");
}

/// Serialized bytes must be stable: re-serializing a deserialized index produces
/// the same byte sequence (no double-normalization or data mutation).
#[test]
fn test_serialize_bytes_stable() {
    let index = build_test_index(200, MetricType::L2);
    let bytes1 = index.serialize_to_bytes().unwrap();
    let restored = HnswIndex::deserialize_from_bytes(&bytes1).unwrap();
    let bytes2 = restored.serialize_to_bytes().unwrap();
    assert_eq!(bytes1, bytes2,
        "re-serializing a deserialized index produced different bytes");
}
```

- [ ] **Step 2: Run tests to verify they compile and pass (establishes baseline)**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo test --test test_hnsw_bulk_io -- --nocapture 2>&1 | tail -20
```

Expected: all 3 tests pass (the serialization format is correct; we're just optimizing its speed, not changing it).

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_hnsw_bulk_io.rs
git commit -m "test: add round-trip serialization tests for bulk I/O fix"
```

---

## Task 2: Fix bulk vector serialization in `write_to`

**Files:**
- Modify: `src/faiss/hnsw.rs:6903-6911`

- [ ] **Step 1: Replace per-f32 vector write with batched write**

Locate this block in `write_to` (around line 6903):
```rust
        // Vectors
        for v in &self.vectors {
            file.write_all(&v.to_le_bytes())?;
        }

        // IDs
        for &id in &self.ids {
            file.write_all(&id.to_le_bytes())?;
        }
```

Replace with:
```rust
        // Vectors — batched write: 65536 f32 per call (256KB), down from 134M individual calls
        {
            const BATCH: usize = 65536;
            let mut byte_buf = vec![0u8; BATCH * 4];
            for chunk in self.vectors.chunks(BATCH) {
                let buf = &mut byte_buf[..chunk.len() * 4];
                for (i, &v) in chunk.iter().enumerate() {
                    buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
                }
                file.write_all(buf)?;
            }
        }

        // IDs — batched write: 32768 i64 per call (256KB)
        {
            const BATCH: usize = 32768;
            let mut byte_buf = vec![0u8; BATCH * 8];
            for chunk in self.ids.chunks(BATCH) {
                let buf = &mut byte_buf[..chunk.len() * 8];
                for (i, &id) in chunk.iter().enumerate() {
                    buf[i * 8..i * 8 + 8].copy_from_slice(&id.to_le_bytes());
                }
                file.write_all(buf)?;
            }
        }
```

- [ ] **Step 2: Build to confirm no errors**

```bash
cargo build 2>&1 | grep "^error"
```

Expected: no output (zero errors).

- [ ] **Step 3: Run round-trip tests**

```bash
cargo test --test test_hnsw_bulk_io -- --nocapture 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): batch vector/ID writes in write_to (134M -> 2K write_all calls)"
```

---

## Task 3: Fix bulk vector deserialization in `read_from`

**Files:**
- Modify: `src/faiss/hnsw.rs:7435-7484`

- [ ] **Step 1: Replace per-f32 vector read with batched read**

Locate this block in `read_from` (around line 7434):
```rust
        // Vectors
        self.vectors = vec![0.0f32; total_f32];
        for i in 0..total_f32 {
            let mut buf = [0u8; 4];
            file.read_exact(&mut buf)?;
            self.vectors[i] = f32::from_le_bytes(buf);
        }
```

Replace with:
```rust
        // Vectors — batched read: 65536 f32 per call (256KB), down from 134M individual calls
        self.vectors = vec![0.0f32; total_f32];
        {
            const BATCH: usize = 65536;
            let mut byte_buf = vec![0u8; BATCH * 4];
            let mut offset = 0;
            while offset < total_f32 {
                let count = BATCH.min(total_f32 - offset);
                let buf = &mut byte_buf[..count * 4];
                file.read_exact(buf)?;
                for i in 0..count {
                    self.vectors[offset + i] =
                        f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
                }
                offset += count;
            }
        }
```

- [ ] **Step 2: Replace per-ID read with batched read**

Locate this block in `read_from` (around line 7464):
```rust
        // IDs
        self.ids = Vec::with_capacity(count);
        let mut id_set = HashSet::with_capacity(count);
        let mut using_sequential_ids = true;
        // OPT-021: Removed HashMap - IDs are stored in order
        for i in 0..count {
            let mut buf = [0u8; 8];
            file.read_exact(&mut buf)?;
            let id = i64::from_le_bytes(buf);
            if !id_set.insert(id) {
                return Err(crate::api::KnowhereError::Codec(format!(
                    "duplicate id in index file: {}",
                    id
                )));
            }
            if id < 0 || id as usize != i {
                using_sequential_ids = false;
            }
            self.ids.push(id);
            // OPT-021: No HashMap insert needed - idx = position in array
        }
        self.use_sequential_ids = using_sequential_ids;
```

Replace with:
```rust
        // IDs — batched read: 32768 i64 per call (256KB)
        self.ids = Vec::with_capacity(count);
        let mut id_set = HashSet::with_capacity(count);
        let mut using_sequential_ids = true;
        {
            const BATCH: usize = 32768;
            let mut byte_buf = vec![0u8; BATCH * 8];
            let mut loaded = 0;
            while loaded < count {
                let batch = BATCH.min(count - loaded);
                let buf = &mut byte_buf[..batch * 8];
                file.read_exact(buf)?;
                for j in 0..batch {
                    let id = i64::from_le_bytes(
                        buf[j * 8..j * 8 + 8].try_into().unwrap()
                    );
                    let global_i = loaded + j;
                    if !id_set.insert(id) {
                        return Err(crate::api::KnowhereError::Codec(format!(
                            "duplicate id in index file: {}",
                            id
                        )));
                    }
                    if id < 0 || id as usize != global_i {
                        using_sequential_ids = false;
                    }
                    self.ids.push(id);
                }
                loaded += batch;
            }
        }
        self.use_sequential_ids = using_sequential_ids;
```

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | grep "^error"
```

Expected: no output.

- [ ] **Step 4: Run round-trip tests**

```bash
cargo test --test test_hnsw_bulk_io -- --nocapture 2>&1 | tail -15
```

Expected: all tests pass.

- [ ] **Step 5: Verify existing HNSW tests still pass**

```bash
cargo test --test test_hnsw_level_multiplier -- --nocapture 2>&1 | tail -10
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): batch vector/ID reads in read_from (134M -> 2K read_exact calls)"
```

---

## Task 4: Eliminate `to_vec()` heap allocations in `add()` and `add_profiled()`

**Files:**
- Modify: `src/faiss/hnsw.rs:1976-1986` and `src/faiss/hnsw.rs:2059-2067`

- [ ] **Step 1: Fix `add()` — replace 1M per-node allocations with one reusable scratch buffer**

Locate this block in `add()` (around line 1976):
```rust
        let mut scratch = SearchScratch::new();
        for (i, &node_level) in node_levels.iter().enumerate().take(n) {
            let idx = first_new_idx + i;

            // Skip the first node (it's the entry point, no connections needed)
            if idx > 0 {
                let vec_start = idx * self.dim;
                let vec: Vec<f32> = self.vectors[vec_start..vec_start + self.dim].to_vec();
                self.insert_node_with_scratch(idx, &vec, node_level, &mut scratch);
            }
        }
```

Replace with:
```rust
        let mut scratch = SearchScratch::new();
        // Pre-allocate one reusable buffer to avoid 1M heap allocs (one per node).
        // copy_from_slice is a plain memcpy with no allocator call.
        let mut vec_scratch = vec![0.0f32; self.dim];
        for (i, &node_level) in node_levels.iter().enumerate().take(n) {
            let idx = first_new_idx + i;

            // Skip the first node (it's the entry point, no connections needed)
            if idx > 0 {
                let vec_start = idx * self.dim;
                vec_scratch.copy_from_slice(&self.vectors[vec_start..vec_start + self.dim]);
                self.insert_node_with_scratch(idx, &vec_scratch, node_level, &mut scratch);
            }
        }
```

- [ ] **Step 2: Fix `add_profiled()` — same change**

Locate this block in `add_profiled()` (around line 2059):
```rust
        let mut scratch = SearchScratch::new();
        for (i, &node_level) in node_levels.iter().enumerate().take(n) {
            let idx = first_new_idx + i;
            if idx > 0 {
                let vec_start = idx * self.dim;
                let vec: Vec<f32> = self.vectors[vec_start..vec_start + self.dim].to_vec();
                self.insert_node_profiled_with_scratch(idx, &vec, node_level, stats, &mut scratch);
            }
        }
```

Replace with:
```rust
        let mut scratch = SearchScratch::new();
        let mut vec_scratch = vec![0.0f32; self.dim];
        for (i, &node_level) in node_levels.iter().enumerate().take(n) {
            let idx = first_new_idx + i;
            if idx > 0 {
                let vec_start = idx * self.dim;
                vec_scratch.copy_from_slice(&self.vectors[vec_start..vec_start + self.dim]);
                self.insert_node_profiled_with_scratch(idx, &vec_scratch, node_level, stats, &mut scratch);
            }
        }
```

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | grep "^error"
```

Expected: no output.

- [ ] **Step 4: Run round-trip tests + level multiplier tests**

```bash
cargo test --test test_hnsw_bulk_io --test test_hnsw_level_multiplier -- --nocapture 2>&1 | tail -15
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): eliminate per-node to_vec() in add/add_profiled (1M allocs -> 1)"
```

---

## Task 5: Add QPS profiling instrumentation in `knowhere_search`

**Files:**
- Modify: `src/ffi.rs:2384-2441`

- [ ] **Step 1: Add timing + nq logging behind env var**

Locate `knowhere_search` in `src/ffi.rs` (around line 2384):
```rust
pub extern "C" fn knowhere_search(
    index: *const std::ffi::c_void,
    query: *const f32,
    count: usize,
    top_k: usize,
    dim: usize,
) -> *mut CSearchResult {
    if index.is_null() || query.is_null() || count == 0 || top_k == 0 || dim == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let index = &*(index as *const IndexWrapper);

        let query_slice = std::slice::from_raw_parts(query, count * dim);

        let result = if let Some(ref idx) = index.sparse_inverted {
```

Add timing around the search call. Insert `let t0 = std::time::Instant::now();` before the `let result = ...` block and add the trace print after the result is computed. The full modified block:

```rust
pub extern "C" fn knowhere_search(
    index: *const std::ffi::c_void,
    query: *const f32,
    count: usize,
    top_k: usize,
    dim: usize,
) -> *mut CSearchResult {
    if index.is_null() || query.is_null() || count == 0 || top_k == 0 || dim == 0 {
        return std::ptr::null_mut();
    }

    unsafe {
        let index = &*(index as *const IndexWrapper);

        let query_slice = std::slice::from_raw_parts(query, count * dim);

        let t0 = std::time::Instant::now();

        let result = if let Some(ref idx) = index.sparse_inverted {
            index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else if let Some(ref idx) = index.sparse_wand {
            index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else if let Some(ref idx) = index.sparse_wand_cc {
            index.search_sparse_queries(query_slice, dim, |sparse_query| {
                idx.search(sparse_query, top_k, None)
            })
        } else {
            index.search(query_slice, top_k)
        };

        if std::env::var_os("KNOWHERE_RS_TRACE_SEARCH").is_some() {
            eprintln!(
                "TRACE_SEARCH nq={} elapsed_us={}",
                count,
                t0.elapsed().as_micros()
            );
        }

        match result {
            Ok(result) => {
                // ... rest unchanged ...
```

Keep the `match result { Ok(result) => { ... } Err(_) => ... }` block identical to current code.

- [ ] **Step 2: Build**

```bash
cargo build 2>&1 | grep "^error"
```

Expected: no output.

- [ ] **Step 3: Smoke-test the tracing locally**

```bash
KNOWHERE_RS_TRACE_SEARCH=1 cargo run --example benchmark --release 2>&1 | grep "TRACE_SEARCH" | head -5
```

Expected output (values will vary):
```
TRACE_SEARCH nq=1 elapsed_us=312
TRACE_SEARCH nq=1 elapsed_us=298
...
```

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "feat(ffi): add optional search tracing via KNOWHERE_RS_TRACE_SEARCH env var"
```

---

## Task 6: x86 authority benchmark

This task is for Codex (remote execution on hannsdb-x86). Claude reviews the results.

**Files:** None modified. This is a remote run task.

- [ ] **Step 1: Sync updated RS to x86**

```bash
# On hannsdb-x86:
cd /data/work/milvus-rs-integ/knowhere-rs
git pull origin main
```

- [ ] **Step 2: Rebuild libknowhere_rs.so on x86**

```bash
# On hannsdb-x86:
CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
    ~/.cargo/bin/cargo build --release \
    --manifest-path /data/work/milvus-rs-integ/knowhere-rs/Cargo.toml \
    2>&1 | tail -5
```

Expected: `Compiling knowhere_rs ...` then `Finished release [optimized]`.

- [ ] **Step 3: Restart Milvus with knowhere-rs shim**

```bash
# On hannsdb-x86:
bash /data/work/milvus-rs-integ/knowhere-rs/scripts/knowhere-rs-shim/start_standalone_remote.sh
```

Wait 10s for Milvus to initialize. Check:
```bash
curl -s http://localhost:9091/healthz
```
Expected: `{"status":"healthy"}` or similar.

- [ ] **Step 4: Run VectorDBBench Cohere-1M HNSW**

```bash
# On hannsdb-x86:
cd /data/work/VectorDBBench
KNOWHERE_RS_FFI_ENABLE_PARALLEL_HNSW_ADD=1 \
    python run.py \
    --case Performance768D1M \
    --db Milvus \
    --index HNSW \
    --m 16 --ef-construction 128 --ef-search 128 \
    --top-k 100 \
    2>&1 | tee /tmp/vdbbench_round1_$(date +%Y%m%d_%H%M).log
```

Wait for completion (~30-45 min). Monitor with:
```bash
tail -f /tmp/vdbbench_round1_*.log
```

- [ ] **Step 5: Extract and record results**

From the VectorDBBench output, note:
- `insert_duration`
- `optimize_duration`
- `load_duration`
- `qps` (best across concurrency levels)
- `recall`

Write results to `/tmp/codex_status.txt`:
```
DONE: insert=Xs optimize=Xs load=Xs qps=X recall=X.XXX
```

---

## Task 7: Archive results and update status doc

**Files:**
- Create: `benchmark_results/milvus_cohere1m_hnsw_rs_round1_YYYYMMDD.json`
- Modify: `docs/milvus-vectordbbench-native-vs-rs-status.md`

- [ ] **Step 1: Create benchmark result JSON**

Fill in actual numbers from Task 6. File path: `benchmark_results/milvus_cohere1m_hnsw_rs_round1_20260405.json` (use actual date).

```json
{
  "benchmark": "milvus-vectordbbench-cohere1m-hnsw-round1-optimize",
  "date": "2026-04-XX",
  "rs_git_sha": "<git rev-parse HEAD on x86>",
  "params": {
    "case_type": "Performance768D1M",
    "db": "Milvus",
    "index": "HNSW",
    "metric_type": "COSINE",
    "top_k": 100,
    "m": 16,
    "ef_construction": 128,
    "ef_search": 128
  },
  "fixes_applied": [
    "bulk-serialize-deserialize",
    "eliminate-tovec-in-add"
  ],
  "rows": [
    {
      "backend": "milvus-native-knowhere",
      "qps": 848.398,
      "recall": 0.9558,
      "insert_duration": 179.063,
      "optimize_duration": 339.6463,
      "load_duration": 518.7093,
      "source": "baseline-20260405"
    },
    {
      "backend": "milvus-knowhere-rs-round1",
      "qps": <FILL>,
      "recall": <FILL>,
      "insert_duration": <FILL>,
      "optimize_duration": <FILL>,
      "load_duration": <FILL>
    }
  ],
  "ratios": {
    "rs_over_native_qps": <qps_rs / 848.398>,
    "rs_over_native_optimize": <optimize_rs / 339.6463>,
    "rs_over_native_load": <load_rs / 518.7093>
  },
  "verdict": "<rs_below_target | load_load_optimize_beat_native | full_beat_native>"
}
```

- [ ] **Step 2: Add new row to status doc**

In `docs/milvus-vectordbbench-native-vs-rs-status.md`, find Table A and add a new row:

```markdown
| 2026-04-XX | rs-round1 | <insert> | <optimize> | <load> | <qps> | 0.964 | valid | bulk-io + no-tovec |
```

- [ ] **Step 3: Commit**

```bash
git add benchmark_results/milvus_cohere1m_hnsw_rs_round1_*.json
git add docs/milvus-vectordbbench-native-vs-rs-status.md
git commit -m "perf(results): Round 1 benchmark — bulk I/O + no-alloc add

Before: Load=921s Optimize=559s (both slower than native)
After:  Load=<X>s Optimize=<X>s

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

---

## Success Criteria

| Metric | Before | Target | Native |
|--------|--------|--------|--------|
| Load | 921s | < 519s | 519s |
| Optimize | 559s | < 340s | 340s |
| Insert | 361s | ≤ 361s (no regression) | 179s |
| QPS | 284 | ≥ 284 (no regression) | 848 |
| Recall | 0.964 | ≥ 0.960 | 0.956 |
