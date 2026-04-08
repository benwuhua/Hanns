# IVF-SQ8 Milvus Concurrent QPS Benchmark Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish RS IVF-SQ8 concurrent QPS authority numbers (c=1..80) on hannsdb-x86, identify performance characteristics vs HNSW, and determine if further optimization is needed.

**Architecture:** Direct pymilvus concurrent benchmark using Python threading (same methodology as VectorDBBench's concurrency sweep). Two scale tiers: 100K (quick, confirm pattern) and 1M (authority).

**Tech Stack:** Python threading, pymilvus, hannsdb-x86

---

## Context

### What We Know

IVF-SQ8 is now wired in Milvus. Serial QPS = 39 (100K×768D) — dominated by Milvus fixed overhead H ≈ 25ms/query. This is expected. The real question is: at c=20/c=80 concurrent queries, does RS IVF-SQ8 batch efficiently?

**HNSW Milvus model**: Uses `HNSW_NQ_POOL` to batch nq queries into a single FFI call. At c=80, Milvus FIFO scheduler batches ~80 queries → 1 FFI call → 1042 QPS.

**IVF-SQ8 model**: No explicit nq batching in current implementation. Each query is a separate FFI call. Under concurrency, Milvus executor dispatches queries to RS via `cgo.Async`. Whether these get batched depends on Milvus scheduling.

**Expected outcomes**:
- If IVF-SQ8 scales linearly with concurrency: c=80 ≈ 80 × c=1 / (H + t_search) → ~300-400 QPS (each query ~2ms search time)
- If scheduling overhead dominates: plateau earlier

### Key Parameters

- Machine: hannsdb-x86 (16 cores, CGO_EXECUTOR_SLOTS=32)
- Dataset: synthetic normalized float32 (same shape as Cohere-1M: 768D, IP metric)
- nlist: 1024 (production-realistic)
- nprobe sweep: 8, 32, 128
- Concurrency: serial, c=1, c=5, c=10, c=20, c=40, c=80

### Key Paths

```
Milvus running: hannsdb-x86:19530
Python: /data/work/VectorDBBench/.venv/bin/python3
Benchmark script: /tmp/ivfsq8_concurrent_bench.py
Results local: /Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/
```

---

## Task 1: Concurrent QPS Benchmark (100K×768D, nlist=1024)

**Purpose**: Confirm the concurrency scaling pattern quickly before running the heavier 1M tier.

### Step 1: Write benchmark script to hannsdb-x86

Write `/tmp/ivfsq8_concurrent_bench.py` via SSH:

```python
import time, threading, numpy as np, sys
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

connections.connect(host="127.0.0.1", port="19530")

N, DIM = 100000, 768
NLIST = 1024
schema = CollectionSchema([
    FieldSchema("id", DataType.INT64, is_primary=True),
    FieldSchema("v", DataType.FLOAT_VECTOR, dim=DIM),
])

np.random.seed(42)
vecs = np.random.randn(N, DIM).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

COL = "ivfsq8_bench_concurrent"
if utility.has_collection(COL):
    utility.drop_collection(COL)
col = Collection(COL, schema)

# Insert
t0 = time.time()
BATCH = 5000
for i in range(0, N, BATCH):
    col.insert([list(range(i, min(i+BATCH, N))), vecs[i:min(i+BATCH,N)].tolist()])
col.flush()
insert_t = time.time() - t0
print(f"Insert: {insert_t:.1f}s ({N} vectors, {DIM}D)")

# Build index
t0 = time.time()
col.create_index("v", {
    "index_type": "IVF_SQ8",
    "metric_type": "IP",
    "params": {"nlist": NLIST}
})
utility.wait_for_index_building_complete(COL)
build_t = time.time() - t0
print(f"Build (nlist={NLIST}): {build_t:.1f}s")

col.load()
time.sleep(5)

np.random.seed(99)
all_queries = np.random.randn(500, DIM).astype(np.float32)
all_queries /= np.linalg.norm(all_queries, axis=1, keepdims=True)

def search_one(q, nprobe):
    col.search(
        [q.tolist()], "v",
        {"metric_type": "IP", "params": {"nprobe": nprobe}},
        limit=10
    )

def run_concurrent(concurrency, nprobe, duration_secs=15):
    """Run concurrent searches for duration_secs, return QPS."""
    count = [0]
    stop = [False]
    errors = [0]

    def worker(qidx):
        while not stop[0]:
            try:
                search_one(all_queries[qidx % len(all_queries)], nprobe)
                count[0] += 1
                qidx += concurrency
            except Exception:
                errors[0] += 1

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.time()
    for t in threads: t.start()
    time.sleep(duration_secs)
    stop[0] = True
    for t in threads: t.join(timeout=5)
    elapsed = time.time() - t0
    qps = count[0] / elapsed
    return qps, errors[0]

print(f"\n{'Concurrency':>12} {'nprobe':>8} {'QPS':>10}")
print("-" * 35)

for nprobe in [8, 32, 128]:
    for c in [1, 5, 10, 20, 40, 80]:
        qps, errs = run_concurrent(c, nprobe, duration_secs=15)
        err_str = f" ({errs} err)" if errs > 0 else ""
        print(f"{c:>12} {nprobe:>8} {qps:>10.1f}{err_str}", flush=True)

utility.drop_collection(COL)
print("\nDONE")
```

### Step 2: Run the benchmark

```bash
ssh hannsdb-x86 'timeout 900 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_concurrent_bench.py 2>&1 | tee /tmp/ivfsq8_concurrent_results.txt'
```

This takes about 10-15 minutes (15s × 3 nprobe × 6 concurrency = 270s measurement + overhead).

### Step 3: Retrieve results

```bash
ssh hannsdb-x86 'cat /tmp/ivfsq8_concurrent_results.txt'
```

### Step 4: Record results to local file

Save raw output to `/Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/ivf_sq8_milvus_rs_concurrent_2026-04-08.md`

---

## Task 2: Analysis & wiki ingest

**Purpose**: Interpret results, compare with HNSW model, determine if optimization is needed.

### Step 1: Analyze scaling pattern

From the results, check:
- Does QPS scale near-linearly from c=1 to c=20? (good: IVF-SQ8 compute limited)
- Does it plateau early (c<10)? (bad: overhead limited, like serial H model)
- Does nprobe affect QPS at high concurrency? (yes = compute limited, no = overhead limited)

**Expected**: At c=20+, nprobe should start mattering (more compute per query). If not, it means Milvus overhead still dominates even at high concurrency.

### Step 2: Compare with HNSW

| Metric | HNSW R8 | IVF-SQ8 (expected) |
|--------|---------|---------------------|
| c=1 serial | 111 QPS | ~39 QPS (H dominated) |
| c=20 | 1051 QPS | ? |
| c=80 | 1042 QPS | ? |

HNSW achieves 1042 QPS because the HNSW_NQ_POOL batches 80 queries into a single FFI call (nq=80 → 1 overhead H). IVF-SQ8 has no equivalent batching — expect lower c=80 QPS unless Milvus naturally batches queries.

### Step 3: Update wiki

Update `wiki/benchmarks/authority-numbers.md` with IVF-SQ8 Milvus concurrent results.
Update `wiki/log.md` with a new entry.

If results show poor scaling (c=80 QPS < 200): note that nq-batching optimization is needed (similar to HNSW R7).
If results show good scaling (c=80 QPS > 500): note RS IVF-SQ8 Milvus integration is production-ready.

---

## Success Criteria

1. Concurrent QPS results table collected (3 nprobe × 6 concurrency = 18 data points)
2. Scaling pattern identified (linear/plateau/saturating)
3. Wiki updated with results
4. Recommendation written: needs optimization OR production-ready

## Anti-patterns to Avoid

- Do NOT run the 1M tier unless 100K tier is confirmed working (save time)
- Do NOT compare with Cohere recall (synthetic data recall is not meaningful)
- Do NOT skip the full concurrency sweep (plateau point is the key info)
