# HNSW nq Thread Pool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Parallelize nq queries within each HNSW segment search call using a global, size-controlled thread pool — matching native knowhere C++'s `ThreadPool::GetGlobalThreadPool()` pattern to close the RS vs native 2.3× QPS gap at c=80.

**Architecture:** Native knowhere C++ submits one task per nq query to a shared `folly::CPUThreadPoolExecutor`, then blocks with `fut.wait()` (calling thread de-scheduled, not a worker). RS currently processes nq serially. Our fix: a global `rayon::ThreadPool` sized to `hardware_concurrency - CGO_EXECUTOR_SLOTS` (leaving CPU headroom for CGO threads that temporarily join the pool via `install()`). All concurrent segment FFI calls share the same pool, so total active threads ≈ hardware_concurrency.

**Tech Stack:** Rust, rayon (already a dependency), `once_cell::sync::Lazy` (already in Cargo.toml), `std::sync::Mutex`

---

## Why rayon par_iter failed (context for implementer)

At c=80 Milvus runs 32 concurrent segment FFI calls (one per segment). When each FFI call uses `into_par_iter()`, the calling thread JOINS the rayon global pool (work-steals). Result: 32 CGO threads + N rayon threads = N+32 threads on N CPU cores → severe context switching → QPS drops 349→71.

Native uses `pool_->push()` + `fut.wait()` where `fut.wait()` truly blocks (de-schedules) the calling thread. Pool workers run uncontested.

Our fix: `HNSW_NQ_POOL.install(|| rayon::scope_fifo(...))` where HNSW_NQ_POOL is sized so pool_threads + expected_injecting_callers ≈ hardware_concurrency.

---

## Files

- Modify: `src/faiss/hnsw.rs` — add `HNSW_NQ_POOL` static, replace nq loops in `search_with_bitset_ref` (line 6723) and `search_with_bitset` (line 4352)
- Test: `src/faiss/hnsw.rs` — existing `test_hnsw_search_with_bitset` (line 11128) covers single-query; add multi-query correctness test

---

### Task 1: Add multi-query correctness test (TDD — write test first)

**Files:**
- Modify: `src/faiss/hnsw.rs` — add test near line 11128

- [ ] **Step 1: Write failing test that exercises nq > NQ_PARALLEL_THRESHOLD**

Add this test after `test_hnsw_search_with_bitset` (around line 11200 in the `#[cfg(test)]` block):

```rust
#[test]
fn test_hnsw_search_with_bitset_ref_multi_query() {
    use crate::bitset::BitsetView;

    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: crate::api::DataType::Float,
        params: crate::api::IndexParams::default(),
    };

    let mut index = HnswIndex::new(&config).unwrap();

    // Build index with 20 vectors so ef=5 searches work
    let mut vectors: Vec<f32> = Vec::new();
    for i in 0..20 {
        vectors.extend_from_slice(&[i as f32, 0.0, 0.0, 0.0]);
    }
    let ids: Vec<i64> = (0..20).collect();
    index.train(&vectors).unwrap();
    index.add(&vectors, Some(&ids)).unwrap();

    let req = SearchRequest {
        top_k: 3,
        nprobe: 10,
        filter: None,
        params: None,
        radius: None,
    };

    let empty_bitset = BitsetView::new(std::ptr::null(), 0);
    let bitset_ref = crate::bitset::BitsetRef::from_bitset_view(&empty_bitset);

    // 8 queries — exceeds NQ_PARALLEL_THRESHOLD=4, exercises parallel path
    let mut query_batch: Vec<f32> = Vec::new();
    let expected_nearest: Vec<i64> = vec![0, 1, 2, 3, 4, 5, 6, 7]; // nearest to each query
    for i in 0..8usize {
        query_batch.extend_from_slice(&[i as f32 + 0.1, 0.0, 0.0, 0.0]);
    }

    let result = index.search_with_bitset_ref(&query_batch, &req, &bitset_ref).unwrap();

    // 8 queries × top_k=3 = 24 results
    assert_eq!(result.ids.len(), 24, "expected 8×3=24 ids");
    assert_eq!(result.distances.len(), 24);

    // Each query's top-1 result should be its nearest neighbor
    for q in 0..8 {
        let top1_id = result.ids[q * 3];
        assert_eq!(
            top1_id, expected_nearest[q],
            "query {q}: expected nearest={}, got {}",
            expected_nearest[q], top1_id
        );
    }

    // Results must be ordered nearest-first within each query slot
    for q in 0..8 {
        let d0 = result.distances[q * 3];
        let d1 = result.distances[q * 3 + 1];
        assert!(d0 <= d1, "query {q}: distances not ordered: {d0} > {d1}");
    }
}
```

- [ ] **Step 2: Run test — verify it PASSES on the serial path (baseline)**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo test test_hnsw_search_with_bitset_ref_multi_query -- --nocapture 2>&1 | tail -5
```

Expected: `test test_hnsw_search_with_bitset_ref_multi_query ... ok`

This confirms the test is valid before we change any search logic.

- [ ] **Step 3: Commit the test**

```bash
git add src/faiss/hnsw.rs
git commit -m "test(hnsw): multi-query correctness test for nq >= NQ_PARALLEL_THRESHOLD"
```

---

### Task 2: Implement HNSW_NQ_POOL and parallel search_with_bitset_ref

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add HNSW_NQ_POOL static near top of file (after existing thread_local! block, around line 85)**

Add after the existing `thread_local!` block (around line 84):

```rust
/// Global thread pool for nq-parallel HNSW search.
///
/// Sized to `hardware_concurrency - CGO_EXECUTOR_SLOTS` so that when the
/// `CGO_EXECUTOR_SLOTS` CGO executor threads temporarily join the pool via
/// `install()`, total active threads ≈ hardware_concurrency.
///
/// Mirrors native knowhere's `ThreadPool::GetGlobalThreadPool()` pattern:
/// segment FFI callers submit work here and participate in work-stealing,
/// de-scheduling themselves from the CPU while pool tasks run.
///
/// CGO_EXECUTOR_SLOTS = ceil(maxReadConcurrency × cgoPoolSizeRatio)
///                    = ceil(16 × 2.0) = 32  (from Milvus default config)
const CGO_EXECUTOR_SLOTS: usize = 32;

/// Minimum nq to use the pool; below this rayon task overhead dominates.
const NQ_PARALLEL_THRESHOLD: usize = 4;

static HNSW_NQ_POOL: once_cell::sync::Lazy<rayon::ThreadPool> =
    once_cell::sync::Lazy::new(|| {
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        // Reserve CPU slots for CGO executor threads that will inject via install()
        let pool_threads = hw.saturating_sub(CGO_EXECUTOR_SLOTS).max(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .thread_name(|i| format!("hnsw-nq-{i}"))
            .build()
            .expect("failed to build HNSW_NQ_POOL")
    });
```

- [ ] **Step 2: Replace nq loop in `search_with_bitset_ref` (around line 6753)**

Find this block in `search_with_bitset_ref` (after the `let k = req.top_k;` / `let ef = ...` lines):

```rust
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
```

Replace with:

```rust
        let mut all_ids = Vec::with_capacity(n_queries * k);
        let mut all_dists = Vec::with_capacity(n_queries * k);

        if n_queries >= NQ_PARALLEL_THRESHOLD {
            // Parallel path: submit nq tasks to HNSW_NQ_POOL.
            // install() makes this calling thread temporarily join the pool
            // (work-stealing), so pool_size + calling_threads ≈ hardware_concurrency.
            // Mutex per slot avoids unsafe; lock contention is negligible (one store per task).
            let per_query: Vec<std::sync::Mutex<Vec<(i64, f32)>>> =
                (0..n_queries).map(|_| std::sync::Mutex::new(vec![])).collect();

            HNSW_NQ_POOL.install(|| {
                rayon::scope_fifo(|s| {
                    for q_idx in 0..n_queries {
                        s.spawn_fifo(move |_| {
                            let q_start = q_idx * self.dim;
                            let query_vec = &query[q_start..q_start + self.dim];
                            let mut res = self.search_single_with_bitset_ref(query_vec, ef, k, bitset);
                            res.truncate(k);
                            *per_query[q_idx].lock().unwrap() = res;
                        });
                    }
                });
            });

            for slot in per_query {
                for (id, dist) in slot.into_inner().unwrap() {
                    all_ids.push(id);
                    all_dists.push(dist);
                }
            }
        } else {
            for q_idx in 0..n_queries {
                let q_start = q_idx * self.dim;
                let query_vec = &query[q_start..q_start + self.dim];
                let results = self.search_single_with_bitset_ref(query_vec, ef, k, bitset);
                for (id, dist) in results.into_iter().take(k) {
                    all_ids.push(id);
                    all_dists.push(dist);
                }
            }
        }
```

- [ ] **Step 3: Replace nq loop in `search_with_bitset` (around line 4352)**

Find this block in `search_with_bitset` (after `let should_bruteforce = ...`):

```rust
        for q_idx in 0..n_queries {
            let q_start = q_idx * self.dim;
            let query_vec = &query[q_start..q_start + self.dim];

            let mut results = if should_bruteforce {
                self.brute_force_search(query_vec, k, |_id, idx| {
                    idx >= bitset.len() || !bitset.get(idx)
                })
            } else {
                self.search_single_with_bitset(query_vec, ef, k, bitset)
            };
            if should_bruteforce {
                self.rerank_sq_results(query_vec, &mut results);
            }

            for (id, dist) in results.into_iter().take(k) {
                all_ids.push(id);
                all_dists.push(dist);
            }
        }
```

Replace with:

```rust
        if n_queries >= NQ_PARALLEL_THRESHOLD {
            let per_query: Vec<std::sync::Mutex<Vec<(i64, f32)>>> =
                (0..n_queries).map(|_| std::sync::Mutex::new(vec![])).collect();

            HNSW_NQ_POOL.install(|| {
                rayon::scope_fifo(|s| {
                    for q_idx in 0..n_queries {
                        s.spawn_fifo(move |_| {
                            let q_start = q_idx * self.dim;
                            let query_vec = &query[q_start..q_start + self.dim];
                            let mut results = if should_bruteforce {
                                self.brute_force_search(query_vec, k, |_id, idx| {
                                    idx >= bitset.len() || !bitset.get(idx)
                                })
                            } else {
                                self.search_single_with_bitset(query_vec, ef, k, bitset)
                            };
                            if should_bruteforce {
                                self.rerank_sq_results(query_vec, &mut results);
                            }
                            results.truncate(k);
                            *per_query[q_idx].lock().unwrap() = results;
                        });
                    }
                });
            });

            for slot in per_query {
                for (id, dist) in slot.into_inner().unwrap() {
                    all_ids.push(id);
                    all_dists.push(dist);
                }
            }
        } else {
            for q_idx in 0..n_queries {
                let q_start = q_idx * self.dim;
                let query_vec = &query[q_start..q_start + self.dim];

                let mut results = if should_bruteforce {
                    self.brute_force_search(query_vec, k, |_id, idx| {
                        idx >= bitset.len() || !bitset.get(idx)
                    })
                } else {
                    self.search_single_with_bitset(query_vec, ef, k, bitset)
                };
                if should_bruteforce {
                    self.rerank_sq_results(query_vec, &mut results);
                }

                for (id, dist) in results.into_iter().take(k) {
                    all_ids.push(id);
                    all_dists.push(dist);
                }
            }
        }
```

Also add `let mut all_ids = Vec::with_capacity(n_queries * k);` and `let mut all_dists = Vec::with_capacity(n_queries * k);` just before this block (replacing the existing `Vec::new()` declarations).

- [ ] **Step 4: Build and check for errors**

```bash
cargo build 2>&1 | grep "^error" | head -20
```

Expected: zero errors.

**If compilation fails with lifetime errors on `self`, `query`, or `bitset` in the scope_fifo closures:** The borrow checker may not accept shared borrows across `scope_fifo`. Fix by changing `s.spawn_fifo(move |_| { ... })` to `s.spawn_fifo(|_| { ... })` (drop the `move` so closure borrows rather than moves). If that also fails, annotate with explicit lifetimes or check if `BitsetRef` is `Sync`.

**If compilation fails because `HnswIndex` is not `Sync`:** Add this near the struct definition:
```rust
// SAFETY: All mutable state during search is thread-local (TLS).
// Index data (vectors, layer0, links) is read-only after build.
unsafe impl Sync for HnswIndex {}
```

- [ ] **Step 5: Run multi-query correctness test**

```bash
cargo test test_hnsw_search_with_bitset_ref_multi_query -- --nocapture 2>&1 | tail -5
```

Expected: `test test_hnsw_search_with_bitset_ref_multi_query ... ok`

- [ ] **Step 6: Run full hnsw test suite**

```bash
cargo test hnsw 2>&1 | tail -8
```

Expected: all pass, zero failures.

- [ ] **Step 7: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): HNSW_NQ_POOL — global pool mirrors native knowhere ThreadPool

Native knowhere C++ parallelizes nq via ThreadPool::GetGlobalThreadPool()
(folly CPUThreadPoolExecutor). Calling thread blocks (de-scheduled) while
pool workers process queries. RS processed nq serially → 349 QPS at c=80
vs native 800+.

Fix: HNSW_NQ_POOL sized to (hardware_concurrency - CGO_EXECUTOR_SLOTS).
CGO executor threads join pool via install() (work-stealing), so:
  pool_threads + injecting_callers ≈ hardware_concurrency (no over-sub)

Uses rayon::scope_fifo for lifetime-safe nq dispatch; Mutex per result
slot (one store/query, ~50ns overhead, negligible vs 2.77ms search).

Applied to both search_with_bitset_ref (Milvus main path) and
search_with_bitset (fallback). Serial path preserved for nq < 4."
```

---

### Task 3: x86 benchmark — verify QPS at c=20 and c=80

**Files:** None (remote execution only)

- [ ] **Step 1: Sync code to hannsdb-x86**

```bash
rsync -av --delete \
  --exclude='.git' \
  --exclude='target' \
  /Users/ryan/.openclaw/workspace-builder/knowhere-rs/ \
  hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/
```

- [ ] **Step 2: Build release on x86**

```bash
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ/knowhere-rs && \
  CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  ~/.cargo/bin/cargo build --release 2>&1 | tail -3"
```

Expected: `Finished release [optimized] target(s)`

- [ ] **Step 3: Restart Milvus to pick up new libknowhere_rs.so**

```bash
ssh hannsdb-x86 "pkill -f 'milvus run standalone' || true; sleep 5; \
  cd /data/work/milvus-rs-integ/milvus-src && \
  nohup bash scripts/start_standalone.sh > /tmp/milvus_restart.log 2>&1 &"
sleep 30
```

- [ ] **Step 4: Reload collection (after Milvus restart it needs reload)**

```bash
ssh hannsdb-x86 "cd /data/work/VectorDBBench && .venv/bin/python3 -c \"
from pymilvus import connections, Collection
connections.connect(host='127.0.0.1', port='19530')
col = Collection('VDBBench')
col.load()
print('Load started')
\""
sleep 120
```

- [ ] **Step 5: Run concurrent sweep c=20,80**

```bash
ssh hannsdb-x86 "cd /data/work/VectorDBBench && \
  nohup .venv/bin/python3 - > /tmp/vdbbench_nq_pool.log 2>&1 << 'PY'
from vectordb_bench.backend.clients.milvus.cli import MilvusHNSW
MilvusHNSW.main(args=[
    '--db-label','milvus-rs-nq-pool',
    '--uri','http://127.0.0.1:19530',
    '--user-name','root',
    '--password','Milvus',
    '--case-type','Performance768D1M',
    '--m','16',
    '--ef-construction','128',
    '--ef-search','128',
    '--skip-drop-old',
    '--skip-load',
    '--skip-search-serial',
    '--search-concurrent',
    '--num-concurrency','20,80',
    '--concurrency-duration','60',
    '--concurrency-timeout','3600',
    '--task-label','rs-nq-pool-test',
], standalone_mode=False)
PY
"
```

Wait 3-4 minutes for completion.

- [ ] **Step 6: Read results**

```bash
ssh hannsdb-x86 "cat \$(ls -t /data/work/VectorDBBench/vectordb_bench/results/Milvus/*.json | head -1) \
  | python3 -m json.tool | grep -E 'qps|concurren' | head -10"
```

Expected (target):
- c=20: significantly higher than previous 91 QPS (target: 300+)
- c=80: approaching native 800+ QPS (target: 600+)

**If c=80 QPS is still low (< 300):** The pool sizing may be off. Check `hardware_concurrency` on the server:
```bash
ssh hannsdb-x86 "python3 -c 'import os; print(os.cpu_count())'"
```
If < 40 cores, increase POOL_RESERVE or reduce `CGO_EXECUTOR_SLOTS`.

**If c=80 QPS regresses below 349 (worse than serial):** Same over-subscription issue as R6. The `install()` approach may still cause over-subscription. In that case, STOP and report — do not attempt another fix without architectural discussion.

- [ ] **Step 7: Commit benchmark results**

```bash
git commit --allow-empty -m "bench(milvus): HNSW_NQ_POOL c=80 results — <FILL IN ACTUAL NUMBERS>

Before: serial nq, c=20≈91 QPS, c=80≈349 QPS (vs native ~800+)
After:  pool nq,   c=20=X QPS,  c=80=Y QPS

HNSW_NQ_POOL: pool_threads=(hw - 32), install()+scope_fifo() pattern."
```

---

### Task 4: Update memory and document findings

**Files:** Memory file (outside repo) + docs

- [ ] **Step 1: Update project_milvus_hnsw_optimization.md**

Add a new section `## R7 (HNSW_NQ_POOL, YYYY-MM-DD)` with:
- The actual c=20 and c=80 QPS numbers from Task 3
- Whether native parity was achieved
- Any residual gaps

- [ ] **Step 2: Update the authority table at the top of the memory file**

If c=80 QPS improved, update the QPS row in the authority table.

- [ ] **Step 3: Commit any doc changes in repo**

```bash
git add docs/ && git commit -m "docs(milvus): update HNSW authority numbers post R7"
```

---

## Expected Outcome

If the model is correct (pool_threads + 32 callers ≈ hardware_concurrency):

```
c=80: 2560 tasks / hardware_concurrency × 2.77ms ≈ 101ms per SearchTask
      + 11ms Milvus overhead = 112ms
      QPS = 80/112ms ≈ 714 ≈ native 800+ range
```

The remaining gap vs native (if any) is due to:
1. Rayon task dispatch overhead vs folly's lock-free semaphore queue
2. Cache effects from parallel random graph traversal
3. Pool sizing imprecision (CGO_EXECUTOR_SLOTS may differ from actual concurrency)
