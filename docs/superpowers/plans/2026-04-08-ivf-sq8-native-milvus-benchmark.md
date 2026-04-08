# IVF-SQ8 Native Milvus Concurrent QPS Benchmark Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Establish native Milvus IVF-SQ8 concurrent QPS baseline (c=1..80, nprobe=8/32/128) on hannsdb-x86, then produce RS vs native side-by-side comparison.

**Architecture:** hannsdb-x86 runs one Milvus instance (RS-shim version). To get native numbers we add a `KNOWHERE_RS_IVFSQ8_BYPASS=1` env var check inside the RS intercept block in `index_factory.h`. When set, the IVF-SQ8 index falls through to native. Rebuild shim → restart Milvus with env var → run same concurrent benchmark → restore. Same Python benchmark script as RS run (already confirmed working).

**Tech Stack:** C++ (index_factory.h patch), CMake rebuild, pymilvus, Python threading, hannsdb-x86

---

## Context

### RS results already collected (2026-04-08)

| Concurrency | nprobe=8 | nprobe=32 | nprobe=128 |
|-------------|----------|-----------|------------|
| c=1  | 40.3 | 16.3 | 16.3 |
| c=5  | 129.7 | 61.2 | 69.8 |
| c=10 | 134.2 | 135.5 | 136.6 |
| c=20 | 139.0 | 138.2 | 138.7 |
| c=40 | 138.6 | 138.4 | 137.8 |
| c=80 | 139.5 | 135.1 | 135.9 |

**Hypothesis**: Native will show the same ~139 QPS ceiling (Milvus dispatch dominates equally for both). If native is higher, RS has an extra per-query overhead worth investigating.

### Key Paths

```
Shim: /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/index_factory.h
Build dir: /data/work/milvus-rs-integ/milvus-src/cmake-build
Milvus binary: /data/work/milvus-rs-integ/milvus-src/bin/milvus
Milvus config: /data/work/milvus-rs-integ/milvus-src/configs/milvus.yaml (or equivalent)
Python: /data/work/VectorDBBench/.venv/bin/python3
Benchmark script: /tmp/ivfsq8_concurrent_bench.py (same as RS run)
Results local: /Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/
```

---

## Task 1: Add env var bypass to index_factory.h

**Purpose**: Allow runtime toggle of RS→native fallthrough without needing two separate Milvus binaries.

### Step 1: Add bypass check to the RS intercept block

Edit `/data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/index_factory.h` lines 374–379.

**Current** (lines 374-379):
```cpp
        if (name == IndexEnum::INDEX_FAISS_IVFSQ8 ||
            name == IndexEnum::INDEX_FAISS_IVFSQ) {
            if constexpr (std::is_same_v<T, float>) {
                return Index<IndexNode>(MakeIvfSq8RustNode());
            }
        }
```

**After** (add env var check):
```cpp
        if (name == IndexEnum::INDEX_FAISS_IVFSQ8 ||
            name == IndexEnum::INDEX_FAISS_IVFSQ) {
            if constexpr (std::is_same_v<T, float>) {
                if (!std::getenv("KNOWHERE_RS_IVFSQ8_BYPASS")) {
                    return Index<IndexNode>(MakeIvfSq8RustNode());
                }
            }
        }
```

Apply via SSH:
```bash
ssh hannsdb-x86 "sed -i 's/                return Index<IndexNode>(MakeIvfSq8RustNode());/                if (!std::getenv(\"KNOWHERE_RS_IVFSQ8_BYPASS\")) {\n                    return Index<IndexNode>(MakeIvfSq8RustNode());\n                }/' /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/index_factory.h"
```

**Verify** the edit looks correct:
```bash
ssh hannsdb-x86 'sed -n "374,382p" /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/index_factory.h'
```

### Step 2: Rebuild the knowhere shim

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/cmake-build && make knowhere -j$(nproc) 2>&1 | tail -20'
```

If `make knowhere` doesn't exist, try:
```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src/cmake-build && make -j$(nproc) 2>&1 | tail -30'
```

Expected: compilation of `index_factory.h` consumers (template-heavy, may recompile several .cpp files). Should complete in 2-5 min.

### Step 3: Restart Milvus with bypass env var

First, stop current Milvus:
```bash
ssh hannsdb-x86 'pkill -f "milvus run standalone" && sleep 5 && echo "stopped"'
```

Start Milvus with bypass env var:
```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src && KNOWHERE_RS_IVFSQ8_BYPASS=1 nohup ./bin/milvus run standalone > /tmp/milvus_native_run.log 2>&1 &'
ssh hannsdb-x86 'sleep 10 && grep -i "ready\|listening\|error" /tmp/milvus_native_run.log | tail -5'
```

Verify Milvus is up:
```bash
ssh hannsdb-x86 '/data/work/VectorDBBench/.venv/bin/python3 -c "from pymilvus import connections; connections.connect(host=\"127.0.0.1\", port=\"19530\"); print(\"OK\")"'
```

---

## Task 2: Run native concurrent benchmark

**Purpose**: Same 18-point sweep (3 nprobe × 6 concurrency) but now hitting native Milvus IVF-SQ8.

### Step 1: Write benchmark script

Write `/tmp/ivfsq8_native_concurrent_bench.py` via SSH (same as RS version, different collection name):

```python
import time, threading, numpy as np
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

COL = "ivfsq8_native_concurrent"
if utility.has_collection(COL):
    utility.drop_collection(COL)
col = Collection(COL, schema)

t0 = time.time()
BATCH = 5000
for i in range(0, N, BATCH):
    col.insert([list(range(i, min(i+BATCH, N))), vecs[i:min(i+BATCH,N)].tolist()])
col.flush()
insert_t = time.time() - t0
print(f"Insert: {insert_t:.1f}s ({N} vectors, {DIM}D)")

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
    return count[0] / elapsed, errors[0]

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
ssh hannsdb-x86 'timeout 900 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_native_concurrent_bench.py 2>&1 | tee /tmp/ivfsq8_native_concurrent_results.txt'
```

~10-15 minutes (same timing as RS run).

### Step 3: Retrieve results

```bash
ssh hannsdb-x86 'cat /tmp/ivfsq8_native_concurrent_results.txt'
```

---

## Task 3: Restore RS Milvus + record results

### Step 1: Stop native Milvus and restore RS

```bash
ssh hannsdb-x86 'pkill -f "milvus run standalone" && sleep 5 && echo "stopped"'
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src && nohup ./bin/milvus run standalone > /tmp/milvus_rs_run.log 2>&1 &'
ssh hannsdb-x86 'sleep 10 && grep -i "ready\|listening\|error" /tmp/milvus_rs_run.log | tail -5'
```

### Step 2: Save results to local file

Save raw output to:
`/Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/ivf_sq8_native_milvus_concurrent_2026-04-08.md`

Format:
```markdown
# IVF-SQ8 Native Milvus Concurrent Benchmark — 2026-04-08

**Dataset**: synthetic float32 normalized, 100K × 768D, IP metric
**Machine**: hannsdb-x86 (16 cores)
**Index**: IVF_SQ8 native (KNOWHERE_RS_IVFSQ8_BYPASS=1), nlist=1024

## Build
| Insert | Build |
...

## QPS Table
| Concurrency | nprobe=8 | nprobe=32 | nprobe=128 |
...
```

---

## Task 4: Analysis + wiki update

### Step 1: Build RS vs native comparison table

Compare each cell: RS QPS vs native QPS. Look for:
- Are ceilings the same (~139 QPS)? → confirms Milvus dispatch dominates equally
- Is native higher at any concurrency level? → RS has extra overhead
- Is native lower? → RS is already faster per-query (surprising but possible)

### Step 2: Update wiki/benchmarks/authority-numbers.md

Add a "RS vs Native" comparison table to the IVF-SQ8 Milvus section.

### Step 3: Update wiki/log.md

Add entry with key findings and recommendation.

---

## Success Criteria

1. Native concurrent QPS table collected (3 nprobe × 6 concurrency = 18 data points)
2. RS vs native comparison complete
3. Scaling pattern confirmed: same ceiling (Milvus dispatch) or RS gap (per-query overhead)
4. Wiki updated
5. RS Milvus restored and functional

## Anti-patterns to Avoid

- Do NOT leave Milvus in native-bypass mode at the end (restore RS)
- Do NOT forget to verify Milvus is up after restart before running benchmark
- Do NOT skip verifying the env var bypass actually took effect (check that native build/search behaves differently, e.g., build time may differ slightly)
