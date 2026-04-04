# Milvus + knowhere-rs vs Native VectorDBBench Status

As of 2026-04-03.

This document summarizes the current single-host authority environment,
benchmark method, measured progress, and remaining problems for comparing:

- Milvus + native knowhere
- Milvus + knowhere-rs

The current authority host is `hannsdb-x86`.

## Scope

This is the Milvus-integrated benchmark track, not the standalone Rust HNSW
benchmark track. The purpose is to compare native Milvus/knowhere against
Milvus with `knowhere-rs` under the same VectorDBBench workload.

The main active debugging lane has been:

- `OpenAI-MEDIUM-500K`, `1536D`, `COSINE`, `HNSW`

An earlier comparison lane also exists for:

- `Cohere-MEDIUM-1M`, `768D`, `COSINE`, `HNSW`

## Authority Environment

All acceptance evidence is taken from the existing remote x86 host only.

### Remote Paths

- Host: `hannsdb-x86`
- Milvus integration repo: `/data/work/milvus-rs-integ/milvus-src`
- knowhere-rs integration checkout: `/data/work/milvus-rs-integ/knowhere-rs`
- Rust target dir: `/data/work/milvus-rs-integ/knowhere-rs-target`
- VectorDBBench repo: `/data/work/VectorDBBench`
- Milvus runtime root: `/data/work/milvus-rs-integ/milvus-var`
- Main Milvus log:
  `/data/work/milvus-rs-integ/milvus-var/logs/standalone-stage1.log`
- VectorDBBench logs:
  `/data/work/VectorDBBench/logs/`
- VectorDBBench result JSONs:
  `/data/work/VectorDBBench/vectordb_bench/results/Milvus/`

### Canonical Startup

Do not hand-roll standalone startup. Use:

```bash
cd /data/work/milvus-rs-integ/milvus-src
scripts/knowhere-rs-shim/start_standalone_remote.sh
```

That wrapper is the canonical entrypoint because it already carries:

- embed etcd
- runtime paths
- library paths
- runtime root under `/data/work/milvus-rs-integ/milvus-var`
- health wait on `http://127.0.0.1:9091/healthz`

### Canonical knowhere-rs Rebuild

```bash
cd /data/work/milvus-rs-integ/knowhere-rs
source "$HOME/.cargo/env" >/dev/null 2>&1 || true
CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target cargo build --release --lib
```

## Benchmark Method

## Fairness Contract

For native vs rs comparisons, keep these fixed:

- same host: `hannsdb-x86`
- same Milvus standalone shape
- same VectorDBBench checkout
- same dataset lane
- same HNSW params
- same query workload
- same result/log collection path

Only the backend is allowed to change:

- native Milvus/knowhere
- Milvus + `knowhere-rs`

## Canonical VectorDBBench Entry Points

### 500k OpenAI

- native: `/data/work/VectorDBBench/run_milvus_hnsw_500k_native.py`
- rs: `/data/work/VectorDBBench/run_milvus_hnsw_500k_rs.py`

### 1M Cohere

- native: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_native.py`
- rs: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`

## Validity Rule for Search Metrics

Not every VectorDBBench result JSON is trustworthy.

`qps/recall` is treated as valid only when search starts after the final large
post-compaction segment has already finished:

- HNSW build
- index save
- QueryNode load

If search overlaps with those background tasks, the run is still useful for:

- `insert_duration`
- `optimize_duration`
- `load_duration`

But its `qps/recall` should be treated as polluted.

## Current Measured Progress

## A. Cohere 1M Comparison

This lane established the first integrated native-vs-rs comparison and exposed
the original `save/load` bottlenecks.

| Lane | Insert (s) | Optimize (s) | Load (s) | QPS | Recall | Status |
|---|---:|---:|---:|---:|---:|---|
| native 1M | 179.0630 | 339.6463 | 518.7093 | 848.3980 | 0.9558 | valid |
| rs old | 120.0815 | 2552.0039 | 2672.0854 | 350.0788 | 0.9594 | valid |
| rs buffered-io rerun | 321.3929 | 914.7687 | 1236.1617 | 352.3533 | 0.9614 | valid |
| rs buffered-io first run | 319.8884 | 333.6131 | 653.5015 | 138.7446 | 0.9602 | invalid search window |

Takeaways:

- rs originally inserted faster than native, but `optimize/load` were far
  slower.
- Buffered I/O materially improved `optimize/load`.
- The first buffered-I/O run had polluted search metrics because search started
  while segment replacement was still in progress.
- Buffered I/O alone did not close the search throughput gap to native.

## B. OpenAI 500k Comparison

This has become the main active debugging lane because it is faster to rerun and
still reproduces the integrated Milvus issues.

### Stable reference rows

| Lane | Insert (s) | Optimize (s) | Load (s) | QPS | Recall | Status |
|---|---:|---:|---:|---:|---:|---|
| native 500k | 269.2442 | 71.9112 | 341.1555 | 420.4177 | 0.9869 | valid |
| rs old baseline 500k | 279.9484 | 77.7299 | 357.6783 | 493.6312 | 0.9855 | valid |
| rs bad run 500k | 204.0473 | 2457.3641 | 2661.4114 | 280.2998 | 0.9690 | valid but regressed |
| rs `flat_graph + mem-deserialize` 500k | 360.3625 | 332.2413 | 692.6038 | 282.6973 | 0.9705 | valid |
| rs `parallel + cap=512` 500k | 335.9905 | 156.8985 | 492.8890 | 102.8409 | 0.9889 | invalid search window |

### What has already improved

#### 1. Build-side improvement

Build-time `layer0_flat_graph` refresh during parallel build improved small
segment HNSW build timings materially. Earlier rs small-segment build clusters
were around `~7.4s-8.4s`; after the layout change, many small-segment build
samples moved back into the native-like `~4s-6s` band.

#### 2. Load-side improvement

HNSW `deserialize_from_bytes()` was changed from:

- write bytes to temporary file
- re-open from disk
- full reread + decode

to direct in-memory deserialization. This removed the earlier multi-minute
QueryNode load tails on large HNSW segments.

Observed effect:

- earlier large compaction segments could take `1m+ ~ 2m30s` to load
- after the in-memory deserialize change, large-segment `loadTextIndexesSpan`
  moved back into seconds-scale behavior

#### 3. FFI default behavior hotfix

HNSW FFI now defaults back to serial `add()`. Parallel HNSW add is opt-in only.

This was done because the old FFI behavior had silently flipped Milvus HNSW
builds into `add_parallel()`, which changed graph construction semantics and was
strongly correlated with bad authority `qps/recall`.

## C. Active 500k Follow-up: Parallel Batch Cap

The current targeted experiment is not "parallel vs serial" in the abstract.
It is:

- keep parallel build available
- clamp parallel batch size
- see whether graph quality recovers enough without paying full serial cost

### Local screen evidence

Deterministic local screen on `12k x 128` showed:

| Mode | Threads | Batch Size | Build (s) | Recall@10 |
|---|---:|---:|---:|---:|
| serial | 1 | 1 | 19.88 | 0.9992 |
| parallel | 2 | 5000 | 0.25 | 0.0773 |
| parallel | 4 | 3000 | 1.03 | 0.2547 |
| parallel | 8 | 1500 | 1.97 | 0.6289 |
| parallel | 16 | 750 | 2.48 | 0.8547 |
| parallel | 32 | 375 | 3.08 | 0.9336 |

Additional cap sweeps:

| Cap | Build (s) | Recall@10 |
|---|---:|---:|
| 375 | ~2.60-2.67 | 0.9336 |
| 256 | ~2.76-2.82 | 0.9664 |
| 128 | ~3.01-3.03 | 0.9844 |
| 64 | ~3.45-3.46 | 0.9922 |

Interpretation:

- the dominant variable is batch size, not thread count
- `add_parallel()` is a hard quality/throughput tradeoff
- large batches are fast but destroy graph quality
- smaller batches recover recall, but move build cost back toward serial

### Current authority status of `cap=128`

Latest observed authority state for:

- log: `/data/work/VectorDBBench/logs/rs_500k_parallel_cap128_20260402.log`

Observed facts:

- `insert_duration = 155.71822588006034`
- search began only after the last large post-compaction segments had finished
  build/save/load
- search reached at least:
  - `concurrency 10`: `qps = 281.7129`
  - `concurrency 30`: `qps = 281.0497`
- the benchmark process was still alive during the last observation, so the
  final JSON row had not yet been written

Interpretation:

- unlike `cap=512`, this run appears to have entered a clean search window
- `cap=128` fixes the worst graph-quality collapse much better than larger caps
- but its search throughput still appears materially below native and below the
  old rs baseline

## D. Search QPS Gap Root Cause Analysis (2026-04-03)

### Observation

| Backend | QPS | Recall | ef_search | Metric | Dim | Dataset |
|---------|-----:|-------:|----------:|--------|-----|---------|
| native (faiss IndexHNSWFlatCosine) | **848** | 0.958 | 128 | cosine | 768 | Cohere 1M |
| RS (HnswIndex slab fast path) | **351** | 0.990 | 128 | cosine | 768 | Cohere 1M |

RS is **2.4x slower** but **3.2% higher recall** at the same ef=128.

### Confirmed: ef_search Passes Correctly

Shim (`hnsw_rust_node.cpp` line 234): calls `knowhere_set_ef_search(handle_, 128)`
before each search. RS `set_ef_search()` sets `self.ef_search = 128`. The FFI
`SearchRequest.nprobe` is hardcoded to 8, but `effective_hnsw_ef_search`
resolves to `max(128, 8, 20) = 128`. **ef=128 is correctly used.**

### Architecture Differences

| Aspect | Native (faiss) | RS (knowhere-rs) |
|--------|---------------|------------------|
| Storage | vectors in continuous `codes[]`, neighbors in separate `hnsw.neighbors[]` | interleaved slab: `[count\|neighbors\|vector]` per node |
| Vector stride | dim × 4B = 3072B (pure vector) | (1 + M×2 + dim) × 4B = 3204B (metadata mixed in) |
| Distance compute | `fvec_inner_product_batch_4` (raw pointers) | `ip_batch_4` (4 slices from slab, bounds-checked) |
| Cosine normalization | Pre-computed `inverse_l2_norms[]` table: `IP × inv_vec_norm × inv_query_norm` | Pre-normalized stored vectors: `1.0 - IP / query_norm` |
| Data layout | `IndexFlatCosine` — vectors packed, norms in separate array | `Layer0Slab` — ~1GB slab >> L3 cache |
| Concurrency | `search_pool->push()` — 1 thread/query from thread pool | FFI sync call — 1 thread/query (Milvus dispatches concurrency) |
| Search termination | `has_next()` only — pops until frontier empty | `has_next()` + early exit when `candidate.dist >= worst_dist` |
| Prefetch | `prefetch_L2(inverse_l2_norms + idx)` for norms | None |

### Why RS Recall Is Higher

RS uses serial `add()` for all Milvus builds. Serial build produces better graph
connectivity than native's build path. At ef=128, a better-connected graph finds
more true nearest neighbors → recall 0.990 vs 0.958.

### Why RS QPS Is Lower (Hypotheses Ranked)

**H1 (most likely): Memory layout / cache behavior**
- RS slab = 970MB, stride 3204B with metadata interleaved
- Native codes = 935MB, stride 3072B, pure vectors
- Both exceed L3 (~30MB on x86), so both are DRAM-bound
- But native's `prefetch_L2` on inverse_l2_norms table may help pipeline memory fetches
- RS has no prefetch — CPU stalls on every `vector_slice_for` call

**H2: Per-distance overhead**
- RS cosine path: `vector_slice_for()` × 4 (slice construction + bounds check) + TLS access for `HNSW_COSINE_QUERY_NORM_TLS`
- Native: raw pointer arithmetic + `prefetch_L2` + simple multiply
- For 768D where each IP dominates compute, overhead should be < 5% — unlikely to explain 2.4x

**H3: Graph quality → more hops**
- RS better graph = more candidate exploration per query
- Higher recall confirms more nodes visited
- But ef=128 caps the candidate pool, so hop count should be similar
- This may explain part of the gap (maybe 10-30%), not all of it

**H4: RS slab vector access pattern**
- Slab neighbors_for() → returns `&[u32]` from slab at offset 1
- Slab vector_ptr_for() → raw pointer into slab at offset 1 + max_neighbors
- Each neighbor expansion reads both neighbors AND vectors from the same slab entry
- No spatial locality benefit — random access anyway

### ef-sweep Results (Cohere 1M, 768D, Cosine, M=16, x86, serial build, 2026-04-03)

Build time: 1272s (serial, 1M vectors, 768D cosine)

| ef | recall@10 | single QPS | batch QPS |
|---:|----------:|-----------:|----------:|
| 16 | 0.7829 | 5623 | 27579 |
| 32 | 0.8523 | 4277 | 20346 |
| 48 | 0.8990 | 3157 | 14961 |
| 64 | 0.9240 | 2507 | 11863 |
| 80 | 0.9407 | 2078 | 9845 |
| **96** | **0.9520** | **1765** | **8360** |
| **112** | **0.9599** | **1551** | **7341** |
| 128 | 0.9660 | 1380 | 6454 |
| 160 | 0.9736 | 1106 | 5120 |
| 200 | 0.9799 | 922 | 4372 |

**Key insight**: RS standalone HNSW is fast (ef=96, recall=0.952 → QPS=1765 single, 8360 batch).
But Milvus integration drops QPS from 1380 (standalone ef=128) to 351 (Milvus ef=128) — **a 3.9x overhead**.

**Root cause relocated**: The QPS gap is NOT in HNSW search algorithm or memory layout.
It is in the **Milvus FFI integration path** (bitset conversion, memory copies, Go↔C bridging).

### Recommended Next Steps

1. **Profile Milvus FFI path**: instrument `knowhere_search_with_bitset` to measure time in bitset conversion vs actual search
2. **Eliminate bitset copy**: `knowhere_search_with_bitset` (ffi.rs:2523) copies `bitset_words.to_vec()` — this allocates on every search call
3. **Compare with native shim**: native shim passes bitset by reference (no copy); RS shim copies bitset data into a Vec then constructs BitsetView
4. **Consider zero-copy bitset**: pass bitset pointer directly without intermediate Vec allocation

## Current Problems

## 1. Core Product Problem: `add_parallel()` Batch Semantics

The main remaining product-level issue is no longer the original load bug.
It is the HNSW parallel build path itself.

Current root cause:

- Milvus HNSW FFI used to auto-select `add_parallel()`
- `add_parallel()` computes neighbors batch-first, then performs graph updates
  later
- nodes inside the same batch do not see each other during neighbor search
- this creates a strong quality/throughput tradeoff

This is now the strongest explanation for the bad integrated `qps/recall`
behavior seen in the Milvus lane.

## 2. Large-Segment Optimize Cost

When HNSW FFI is forced fully serial, correctness improves, but `optimize`
becomes too long on large post-compaction segments because final HNSW build is
also serial.

Observed examples:

- large segments around `713MB` and `935MB` serialized size took many minutes to
  finish build

So the current problem is not simply "use serial everywhere".

## 3. Benchmark Validity Risk

Some VectorDBBench runs still start search before the final compaction segment
is fully built and loaded.

When that happens:

- `insert/optimize/load` can still be informative
- `qps/recall` cannot be treated as authoritative

This is why:

- `parallel + cap=512` is treated as a polluted search run
- current `cap=128` is more valuable because its search phase appears to begin
  only after the final large-segment backlog drains

## 4. Operational Problem: Remote Proxy Stability

The SOCKS5 proxy path has also been an operational hazard.

Known operator-side issues:

- rapid or parallel SSH bursts can trigger proxy-side throttling/timeouts
- long-hanging local SSH/SOCKS processes can consume proxy concurrency
- previous local relay code had a nonblocking send bug and was fixed, but
  remote access still benefits from:
  - one SSH command at a time
  - a gap between SSH calls
  - merged remote probes instead of many short probes

This is an execution friction issue, not the Milvus product root cause, but it
does affect how quickly authority evidence can be collected.

## Current Best Interpretation

As of now:

- the original build-side layout problem has a credible fix
- the original multi-minute load problem has a credible fix
- the main remaining product issue is HNSW `add_parallel()` batch semantics
- the best current path is not unlimited parallel HNSW build
- capped parallel build is promising, but still not yet closed as the final
  answer

## Recommended Reading

- [milvus-vectordbbench-authority-runbook.md](milvus-vectordbbench-authority-runbook.md)
- [task-progress.md](../task-progress.md)
