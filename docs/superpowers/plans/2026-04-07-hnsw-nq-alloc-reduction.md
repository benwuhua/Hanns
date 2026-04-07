# HNSW nq Allocation Reduction Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce per-task allocation overhead in the HNSW nq parallel path to close the remaining RS vs native QPS gap (~540 vs ~800 at c=80).

**Architecture:** Two targeted fixes. (1) `Layer0OrderedResults::to_sorted_pairs` unnecessarily clones the entire BinaryHeap before draining it — change to `&mut self` + `std::mem::take` to drain in-place, eliminating one malloc+memcpy per search call. (2) The parallel path collects results via `Mutex<Vec<(i64,f32)>>` per slot then copies them all into `all_ids`/`all_dists` — replace with a pre-allocated flat `Vec<(i64,f32)>` + unsafe non-overlapping slice writes, eliminating the Mutex + final copy loop. Both changes are preceded by a diagnostic run on x86 to confirm actual hw/pool config and nq/batch values.

**Tech Stack:** Rust, rayon (already in use), `std::sync::Mutex` (being removed), raw pointer slice trick (unsafe, disjoint slices)

---

## Why these two fixes

**Root cause evidence (from code analysis vs native hnsw.cc):**

Native `Search()` (hnsw.cc:117–146):
- Allocates `p_id[k*nq]` + `p_dist[k*nq]` once before the loop
- Each task writes directly to `p_id[idx*k]` — zero per-task allocation
- No post-loop copy

RS R7 (hnsw.rs:6788–6815):
- `per_query: Vec<Mutex<Vec<(i64,f32)>>>` — n_queries Mutex slots upfront
- Each task: `search_single_with_bitset_ref` → returns `Vec<(i64,f32)>` (1 malloc) via `to_sorted_pairs()` which **clones BinaryHeap first** (1 extra malloc+memcpy)
- Post-loop: copies all k×n_queries results from Mutex slots to `all_ids`/`all_dists`

At c=80 with 32 segments × 80 queries = 2560 tasks per SearchTask:
- Extra clone per task: 2560 × ~150ns = 384µs wasted per SearchTask
- Mutex + copy loop: 2560 × ~30ns + k×n_queries copy = adds latency

---

## Files

- Modify: `src/faiss/hnsw.rs`
  - `Layer0OrderedResults::to_sorted_pairs` (line 301) — eliminate clone
  - `search_with_bitset_ref` parallel path (line 6788) — pre-alloc flat buffer
  - `search_with_bitset` parallel path (similar structure) — same fix

---

### Task 1: Diagnostic — verify hw, pool_threads, and actual nq/batch on x86

**Files:**
- Modify: `src/faiss/hnsw.rs` — add two `eprintln!` statements

- [ ] **Step 1: Add hw/pool_threads log to HNSW_NQ_POOL initializer**

Find line 104–108 in `src/faiss/hnsw.rs`:
```rust
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        // Reserve CPU slots for CGO executor threads that will inject via install()
        let pool_threads = hw.saturating_sub(CGO_EXECUTOR_SLOTS).max(4);
```

Replace with:
```rust
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        // Reserve CPU slots for CGO executor threads that will inject via install()
        let pool_threads = hw.saturating_sub(CGO_EXECUTOR_SLOTS).max(4);
        eprintln!("[HNSW_POOL] hw={hw} CGO_EXECUTOR_SLOTS={CGO_EXECUTOR_SLOTS} pool_threads={pool_threads}");
```

- [ ] **Step 2: Add nq log at top of search_with_bitset_ref parallel branch**

Find line 6788 in `src/faiss/hnsw.rs`:
```rust
        if n_queries >= NQ_PARALLEL_THRESHOLD {
```

Replace with:
```rust
        if n_queries >= NQ_PARALLEL_THRESHOLD {
            if n_queries > 1 {
                eprintln!("[HNSW_NQ] n_queries={n_queries} k={k} ef={ef}");
            }
```

- [ ] **Step 3: Build locally**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build 2>&1 | grep "^error" | head -10
```

Expected: zero errors.

- [ ] **Step 4: Sync to x86 and rebuild**

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

- [ ] **Step 5: Restart Milvus and run a quick c=20 test to capture logs**

```bash
ssh hannsdb-x86 "pkill -f 'milvus run standalone' || true; sleep 3; \
  cd /data/work/milvus-rs-integ/milvus-src && \
  nohup bin/milvus run standalone > /tmp/milvus_diag.log 2>&1 &"
sleep 20

ssh hannsdb-x86 "cd /data/work/VectorDBBench && \
  timeout 30 .venv/bin/python3 -c \"
from pymilvus import connections, Collection, utility
connections.connect(host='127.0.0.1', port='19530')
cols = utility.list_collections()
if cols:
    col = Collection(cols[0])
    import numpy as np
    np.random.seed(1)
    q = np.random.randn(8, 768).astype('float32').tolist()
    r = col.search(q, 'vector', {'metric_type':'IP','params':{'ef':128}}, limit=100, output_fields=[])
    print('search ok, got', len(r), 'results')
\" 2>&1"

ssh hannsdb-x86 "grep 'HNSW_POOL\|HNSW_NQ' /tmp/milvus_diag.log | head -20"
```

- [ ] **Step 6: Record the diagnostic output**

Expected output examples:
```
[HNSW_POOL] hw=70 CGO_EXECUTOR_SLOTS=32 pool_threads=38
[HNSW_NQ] n_queries=8 k=100 ef=128
```

Write down the actual `hw` and `pool_threads` values. If `pool_threads < 10`, we need to adjust `CGO_EXECUTOR_SLOTS` in a future task.

- [ ] **Step 7: Commit the diagnostic code**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
git add src/faiss/hnsw.rs
git commit -m "$(cat <<'EOF'
diag(hnsw): add hw/pool_threads and nq/batch diagnostic logging

Temporary: print [HNSW_POOL] hw/pool_threads at pool init and [HNSW_NQ]
n_queries per parallel batch to verify pool sizing on x86.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Fix `to_sorted_pairs` — eliminate BinaryHeap clone

**Files:**
- Modify: `src/faiss/hnsw.rs` — lines 301–311 and line 5761

The current `to_sorted_pairs(&self)` at line 301 calls `self.entries.clone()` before consuming via `into_sorted_vec()`. The `clone()` allocates a new heap of the same size and copies all entries. This is unnecessary: since `scratch` is always `&mut`, we can drain the heap in-place with `std::mem::take`.

- [ ] **Step 1: Write a correctness test for the fix before changing anything**

Add this test in the `#[cfg(test)]` block in `src/faiss/hnsw.rs` (near line 11200+, after the existing hnsw tests):

```rust
#[test]
fn test_layer0_ordered_results_to_sorted_pairs_no_clone() {
    // Verify to_sorted_pairs produces correct output after switching from
    // clone()-based to take()-based drain.
    let mut r = Layer0OrderedResults {
        entries: std::collections::BinaryHeap::new(),
    };
    r.prepare(5);
    // Insert 5 entries with known distances
    use super::*; // bring Layer0PoolEntry, SearchMaxDist, etc. in scope if needed
    // Since Layer0PoolEntry and insert() are private, test via the full search path instead.
    // This test verifies sort order of to_sorted_pairs by comparing single-query result
    // before vs after the change (golden test).
    let config = IndexConfig {
        index_type: IndexType::Hnsw,
        metric_type: MetricType::L2,
        dim: 4,
        data_type: crate::api::DataType::Float,
        params: crate::api::IndexParams::default(),
    };
    let mut index = HnswIndex::new(&config).unwrap();
    let vectors: Vec<f32> = (0..10).flat_map(|i| [i as f32, 0.0, 0.0, 0.0]).collect();
    let ids: Vec<i64> = (0..10).collect();
    index.train(&vectors).unwrap();
    index.add(&vectors, Some(&ids)).unwrap();

    let req = SearchRequest { top_k: 3, nprobe: 10, filter: None, params: None, radius: None };
    let bitset = crate::bitset::BitsetRef::new(&[], 0);
    let query = vec![3.1f32, 0.0, 0.0, 0.0];
    let result = index.search_with_bitset_ref(&query, &req, &bitset).unwrap();

    assert_eq!(result.ids.len(), 3);
    // query=[3.1,0,0,0]; nearest is [3,0,0,0]=id3, then [4,0,0,0]=id4, then [2,0,0,0]=id2
    assert_eq!(result.ids[0], 3, "top-1 should be id=3");
    assert_eq!(result.ids[1], 4, "top-2 should be id=4");
    assert_eq!(result.ids[2], 2, "top-3 should be id=2");
    // distances must be ascending
    assert!(result.distances[0] <= result.distances[1]);
    assert!(result.distances[1] <= result.distances[2]);
}
```

- [ ] **Step 2: Run test to confirm it passes on the current (clone-based) code**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo test test_layer0_ordered_results_to_sorted_pairs_no_clone -- --nocapture 2>&1 | tail -5
```

Expected: `ok`

- [ ] **Step 3: Change `to_sorted_pairs` to drain in-place (eliminate clone)**

Find lines 301–311 in `src/faiss/hnsw.rs`:
```rust
    fn to_sorted_pairs(&self) -> Vec<(usize, f32)> {
        let mut pairs: Vec<(usize, f32)> = self
            .entries
            .clone()
            .into_sorted_vec()
            .into_iter()
            .map(|(SearchMaxDist(dist), idx)| (idx, dist))
            .collect();
        pairs.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        pairs
    }
```

Replace with:
```rust
    fn to_sorted_pairs(&mut self) -> Vec<(usize, f32)> {
        // Drain the heap in-place (no clone). The heap is cleared anyway before
        // next use via prepare(), so taking ownership here is safe.
        let mut pairs: Vec<(usize, f32)> = std::mem::take(&mut self.entries)
            .into_sorted_vec()
            .into_iter()
            .map(|(SearchMaxDist(dist), idx)| (idx, dist))
            .collect();
        pairs.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
        pairs
    }
```

The caller at line 5761 is `scratch.layer0_results.to_sorted_pairs()`. Since `scratch: &mut SearchScratch` already, this compiles without further changes.

- [ ] **Step 4: Build**

```bash
cargo build 2>&1 | grep "^error" | head -20
```

Expected: zero errors. If there is a "cannot borrow as mutable" error, check whether any other callers of `to_sorted_pairs` use `&self` instead of `&mut self`.

- [ ] **Step 5: Run correctness test**

```bash
cargo test test_layer0_ordered_results_to_sorted_pairs_no_clone -- --nocapture 2>&1 | tail -5
```

Expected: `ok` (same results as before — the fix must not change output)

- [ ] **Step 6: Run full hnsw test suite**

```bash
cargo test hnsw 2>&1 | tail -5
```

Expected: all pass, 0 failures.

- [ ] **Step 7: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "$(cat <<'EOF'
perf(hnsw): eliminate BinaryHeap clone in to_sorted_pairs

to_sorted_pairs(&self) called self.entries.clone() before consuming via
into_sorted_vec(). This allocated and copied the entire results heap on
every search call (ef=128 entries, ~1.5KB per clone).

Fix: change to &mut self, drain with std::mem::take. The heap is always
cleared via prepare() before next use, so draining is correct.

2560 tasks/SearchTask × 1 eliminated clone = ~384µs saved per SearchTask.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Replace `Mutex<Vec>` per slot with pre-allocated flat buffer + unsafe slices

**Files:**
- Modify: `src/faiss/hnsw.rs` — parallel paths in `search_with_bitset_ref` (line 6788) and `search_with_bitset` (similar)

This replaces the `Vec<Mutex<Vec<(i64,f32)>>>` + post-loop copy pattern with pre-allocated `all_ids`/`all_dists` written directly from tasks via non-overlapping unsafe slices. Mirrors native's `p_id[idx*k]` + `p_dist[idx*k]` pattern exactly.

**SAFETY invariant:** Each `q_idx` is unique (0..n_queries, spawned once). Slices `[q_idx*k .. (q_idx+1)*k]` are disjoint. The parent scope outlives all tasks (rayon `scope_fifo` guarantees this). Raw pointers are valid for the duration of the scope.

- [ ] **Step 1: Find the parallel path in `search_with_bitset_ref`**

The block to replace is lines 6788–6815 (the `if n_queries >= NQ_PARALLEL_THRESHOLD` block). Read it to confirm the exact content matches what's shown below, then replace.

Find this entire block:
```rust
        if n_queries >= NQ_PARALLEL_THRESHOLD {
            // Parallel path: submit nq tasks to HNSW_NQ_POOL.
            // install() makes this calling thread temporarily join the pool
            // (work-stealing), so pool_size + calling_threads ≈ hardware_concurrency.
            let per_query: Vec<std::sync::Mutex<Vec<(i64, f32)>>> =
                (0..n_queries).map(|_| std::sync::Mutex::new(vec![])).collect();
            let per_query_ref = &per_query;

            HNSW_NQ_POOL.install(|| {
                rayon::scope_fifo(|s| {
                    for q_idx in 0..n_queries {
                        s.spawn_fifo(move |_| {
                            let q_start = q_idx * self.dim;
                            let query_vec = &query[q_start..q_start + self.dim];
                            let mut res = self.search_single_with_bitset_ref(query_vec, ef, k, bitset);
                            res.truncate(k);
                            *per_query_ref[q_idx].lock().unwrap() = res;
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
```

Replace with:
```rust
        if n_queries >= NQ_PARALLEL_THRESHOLD {
            // Parallel path: pre-allocate flat output buffers (mirrors native's
            // p_id[idx*k] / p_dist[idx*k] pattern — zero per-task allocation).
            // Initialized with sentinel values; tasks overwrite their slice.
            all_ids.resize(n_queries * k, -1i64);
            all_dists.resize(n_queries * k, f32::INFINITY);

            // Wrap raw pointers in a Copy+Send newtype.
            // SAFETY: each spawned task writes to a disjoint slice [q_idx*k..(q_idx+1)*k].
            // The pointers are valid for the duration of scope_fifo (all tasks complete
            // before scope_fifo returns, which is before all_ids/all_dists are dropped).
            #[derive(Clone, Copy)]
            struct SendPtr<T>(*mut T);
            unsafe impl<T> Send for SendPtr<T> {}

            let ids_ptr = SendPtr(all_ids.as_mut_ptr());
            let dists_ptr = SendPtr(all_dists.as_mut_ptr());

            HNSW_NQ_POOL.install(|| {
                rayon::scope_fifo(|s| {
                    for q_idx in 0..n_queries {
                        s.spawn_fifo(move |_| {
                            let q_start = q_idx * self.dim;
                            let query_vec = &query[q_start..q_start + self.dim];
                            let res = self.search_single_with_bitset_ref(query_vec, ef, k, bitset);
                            // SAFETY: q_idx is unique per task; slices are disjoint.
                            unsafe {
                                let ids_slice = std::slice::from_raw_parts_mut(
                                    ids_ptr.0.add(q_idx * k), k,
                                );
                                let dists_slice = std::slice::from_raw_parts_mut(
                                    dists_ptr.0.add(q_idx * k), k,
                                );
                                for (i, &(id, dist)) in res.iter().take(k).enumerate() {
                                    ids_slice[i] = id;
                                    dists_slice[i] = dist;
                                }
                            }
                        });
                    }
                });
            });
            // all_ids and all_dists already populated — no copy loop needed.
```

- [ ] **Step 2: Find and apply the same replacement in `search_with_bitset`**

The `search_with_bitset` function has a similar parallel path (with a bruteforce branch). Find the `if n_queries >= NQ_PARALLEL_THRESHOLD` block in it (which uses the same `Mutex<Vec<(i64,f32)>>` pattern).

Find:
```rust
            let per_query: Vec<std::sync::Mutex<Vec<(i64, f32)>>> =
                (0..n_queries).map(|_| std::sync::Mutex::new(vec![])).collect();
            let per_query_ref = &per_query;

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
                            *per_query_ref[q_idx].lock().unwrap() = results;
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
```

Replace with:
```rust
            all_ids.resize(n_queries * k, -1i64);
            all_dists.resize(n_queries * k, f32::INFINITY);

            #[derive(Clone, Copy)]
            struct SendPtr<T>(*mut T);
            unsafe impl<T> Send for SendPtr<T> {}

            let ids_ptr = SendPtr(all_ids.as_mut_ptr());
            let dists_ptr = SendPtr(all_dists.as_mut_ptr());

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
                            // SAFETY: q_idx is unique per task; slices are disjoint.
                            unsafe {
                                let ids_slice = std::slice::from_raw_parts_mut(
                                    ids_ptr.0.add(q_idx * k), k,
                                );
                                let dists_slice = std::slice::from_raw_parts_mut(
                                    dists_ptr.0.add(q_idx * k), k,
                                );
                                for (i, &(id, dist)) in results.iter().take(k).enumerate() {
                                    ids_slice[i] = id;
                                    dists_slice[i] = dist;
                                }
                            }
                        });
                    }
                });
            });
```

Note: `all_ids` and `all_dists` must be pre-declared with `Vec::with_capacity` before the `if` block (they already are in the existing code). Verify they are not resized elsewhere in the function before this block. If they were initialized with `Vec::with_capacity(n_queries * k)`, change them to `Vec::with_capacity(0)` (or just `Vec::new()`) since `resize()` does the real allocation now.

Actually: since `resize` is used instead of `push`, the initial `Vec::with_capacity` is fine to keep (capacity hint). The `resize` call will extend up to `n_queries * k` correctly.

- [ ] **Step 3: Build**

```bash
cargo build 2>&1 | grep "^error" | head -20
```

**If build fails with "cannot infer type for type parameter `T` in `SendPtr`":** Add type annotation: `let ids_ptr: SendPtr<i64> = SendPtr(all_ids.as_mut_ptr());`

**If build fails with lifetime or borrow error:** The `self`, `query`, `bitset`, `ef`, `k` captures in the closure should work since they're already used in the R7 code. If a new error appears, check whether `ids_ptr` or `dists_ptr` need explicit lifetime bounds.

- [ ] **Step 4: Run the multi-query test**

```bash
cargo test test_hnsw_search_with_bitset_ref_multi_query -- --nocapture 2>&1 | tail -5
cargo test test_layer0_ordered_results_to_sorted_pairs_no_clone -- --nocapture 2>&1 | tail -5
```

Expected: both `ok`.

- [ ] **Step 5: Run full hnsw suite**

```bash
cargo test hnsw 2>&1 | tail -8
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "$(cat <<'EOF'
perf(hnsw): pre-allocated flat output buffer — mirror native p_id[idx*k] pattern

Native hnsw.cc allocates p_id[k*nq] + p_dist[k*nq] once before the nq
loop; each task writes directly to p_id[idx*k]. RS R7 used Mutex<Vec>
per slot + a post-loop copy into all_ids/all_dists.

Fix: pre-allocate all_ids/all_dists with resize() before scope_fifo.
Tasks write directly to their non-overlapping slice via raw pointer +
unsafe from_raw_parts_mut (SAFETY: q_idx unique per task = disjoint
slices; scope_fifo ensures pointer validity).

Eliminates: n_queries Mutex allocs, n_queries×k result copies in
post-loop. Applied to both search_with_bitset_ref and search_with_bitset.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: x86 benchmark — measure R8 QPS improvement

**Files:** None (remote execution)

- [ ] **Step 1: Sync to x86**

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

- [ ] **Step 3: Restart Milvus**

```bash
ssh hannsdb-x86 "pkill -f 'milvus run standalone' || true; sleep 5; \
  cd /data/work/milvus-rs-integ/milvus-src && \
  nohup bin/milvus run standalone > /tmp/milvus_r8.log 2>&1 &"
sleep 30
ssh hannsdb-x86 "grep 'HNSW_POOL' /tmp/milvus_r8.log | head -3"
```

Record the `[HNSW_POOL]` line — confirms hw and pool_threads on this machine.

- [ ] **Step 4: Reload collection**

```bash
ssh hannsdb-x86 "cd /data/work/VectorDBBench && .venv/bin/python3 -c \"
from pymilvus import connections, Collection, utility
connections.connect(host='127.0.0.1', port='19530')
cols = utility.list_collections()
if cols:
    Collection(cols[0]).load()
    print('Loading', cols[0])
\""
sleep 90
```

- [ ] **Step 5: Run concurrent search sweep c=20, c=80**

```bash
ssh hannsdb-x86 "cat > /tmp/run_r8_bench.py << 'PYEOF'
import time, threading, numpy as np
from pymilvus import connections, Collection, utility

connections.connect(host='127.0.0.1', port='19530')
cols = utility.list_collections()
col = Collection(cols[0])
np.random.seed(42)
q = np.random.randn(1, 768).astype('float32').tolist()
params = {'metric_type': 'IP', 'params': {'ef': 128}}

# Warmup
for _ in range(20):
    col.search(q, 'vector', params, limit=100, output_fields=[])

def bench(concurrency, duration=60):
    count, stop, lock = [0], threading.Event(), threading.Lock()
    def worker():
        while not stop.is_set():
            col.search(q, 'vector', params, limit=100, output_fields=[])
            with lock: count[0] += 1
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    t0 = time.time()
    for t in threads: t.start()
    time.sleep(duration)
    stop.set()
    for t in threads: t.join()
    qps = count[0] / (time.time() - t0)
    print(f'c={concurrency}: {qps:.1f} QPS')
    return qps

print('Running R8 benchmark...')
qps20 = bench(20)
qps80 = bench(80)
print(f'RESULT: c=20={qps20:.1f} c=80={qps80:.1f}')
PYEOF
"
ssh hannsdb-x86 "nohup .venv/bin/python3 /tmp/run_r8_bench.py > /tmp/r8_results.log 2>&1 &"
```

Wait ~3.5 minutes:
```bash
sleep 210
ssh hannsdb-x86 "cat /tmp/r8_results.log"
```

- [ ] **Step 6: Record results**

Expected format:
```
[HNSW_POOL] hw=XX pool_threads=YY   ← from Milvus log
c=20: XXX.X QPS
c=80: XXX.X QPS
RESULT: c=20=XXX.X c=80=XXX.X
```

**Baseline (R7):** c=20=548 QPS, c=80=540 QPS
**Target:** measurable improvement; if c=80 doesn't improve, the gap is elsewhere

**If no improvement (< 5% over R7):**
The bottleneck is likely not allocations but rayon task overhead vs folly's lock-free semaphore. The gap may be fundamental to rayon's work-stealing design. Report BLOCKED with exact numbers for architectural discussion.

- [ ] **Step 7: Commit results**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
git commit --allow-empty -m "$(cat <<'EOF'
bench(milvus): R8 alloc-reduction results — c=20=X c=80=Y QPS

R7 baseline: c=20=548 c=80=540 QPS
R8 (no-clone + flat buffer): c=20=X c=80=Y QPS

hw=XX pool_threads=YY (from HNSW_POOL diagnostic log)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Replace X and Y with actual numbers.

---

### Task 5: Remove diagnostic logging and update memory

**Files:**
- Modify: `src/faiss/hnsw.rs` — remove two `eprintln!` statements added in Task 1
- Modify: `memory/project_milvus_hnsw_optimization.md` (outside repo)

- [ ] **Step 1: Remove the diagnostic `eprintln!` calls added in Task 1**

In `src/faiss/hnsw.rs` line ~108, remove:
```rust
        eprintln!("[HNSW_POOL] hw={hw} CGO_EXECUTOR_SLOTS={CGO_EXECUTOR_SLOTS} pool_threads={pool_threads}");
```

In `src/faiss/hnsw.rs` line ~6789, remove:
```rust
            if n_queries > 1 {
                eprintln!("[HNSW_NQ] n_queries={n_queries} k={k} ef={ef}");
            }
```

- [ ] **Step 2: Build to confirm clean**

```bash
cargo build 2>&1 | grep "^error" | head -5
```

Expected: zero errors.

- [ ] **Step 3: Update `memory/project_milvus_hnsw_optimization.md`**

Add R8 entry to the round history table:
```
| **R8 (alloc-reduction)** | **<commit>** | **c=20=X, c=80=Y** | Eliminate to_sorted_pairs clone + pre-alloc flat output buffer |
```

Update authority table QPS row if improved. Update "Remaining Gaps" section to reflect findings.

If R8 showed minimal improvement: add a section "R8 conclusion: allocation overhead is not the primary gap source; gap is in rayon task overhead vs folly semaphore."

- [ ] **Step 4: Commit cleanup**

```bash
git add src/faiss/hnsw.rs
git commit -m "$(cat <<'EOF'
chore(hnsw): remove diagnostic logging added in R8 investigation

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Expected Outcome

If allocations are the primary cause of the ~30ms extra per SearchTask:
- R8 should give c=80: 600–700 QPS (vs R7's 540, closer to native's ~800)

If allocations are NOT the primary cause:
- R8 gives < 5% improvement
- Root cause is rayon per-task overhead (~5–18µs/task × 2560 = 13–46ms/SearchTask)
- Next investigation: profile with `perf` on x86 to quantify exact per-task overhead

Either outcome is valuable diagnostic information.
