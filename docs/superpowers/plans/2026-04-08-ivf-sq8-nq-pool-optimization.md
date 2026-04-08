# IVF-SQ8 nq 并行优化计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 打破 IVF-SQ8 Milvus 140 QPS 天花板，目标 500+ QPS @ c=80，参照 HNSW R7 的 IVF_NQ_POOL 方案。

**Architecture:** 先诊断：Milvus 到底送几个 nq 给 IVF-SQ8？用现有 `KNOWHERE_RS_TRACE_SEARCH` env var 直接观测。若 nq>1：在 `ivf.rs` search() 加 rayon 并行（镜像 HNSW_NQ_POOL 模式）。若 nq=1：在 C++ shim 层实现请求合批（更复杂，另开计划）。

**Tech Stack:** Rust (rayon, once_cell), hannsdb-x86

---

## Context

### HNSW R7 模型（已验证有效）

```rust
static HNSW_NQ_POOL: once_cell::sync::Lazy<rayon::ThreadPool> = ...;
// CGO executor 线程通过 install() 加入私有 pool → scope_fifo 并行处理 nq 个查询
HNSW_NQ_POOL.install(|| {
    rayon::scope_fifo(|s| {
        for q_idx in 0..n_queries {
            s.spawn_fifo(move |_| { /* search one query, write to disjoint slice */ });
        }
    });
});
```

HNSW c=20: 91→548 QPS (+6×)，c=80: 349→540 QPS（R7），1042 QPS（R8 alloc 优化后）。

### IVF-SQ8 现状

- `ivf.rs` search(): 顺序 for qi in 0..nq 循环
- `ffi.rs` knowhere_search(): 有 `KNOWHERE_RS_TRACE_SEARCH` env var 可记录 nq
- 140 QPS 平台期 = RS parity with native → 瓶颈在 Milvus 层，不在 RS 实现

### Key Paths

```
Source: /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss/ivf.rs
FFI: /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/ffi.rs
Shim C++: hannsdb-x86:/data/work/milvus-rs-integ/.../knowhere-rs-shim/src/ivf_sq8_rust_node.cpp
Benchmark script: /tmp/ivfsq8_concurrent_bench.py (已存在)
Results: /Users/ryan/.openclaw/workspace-builder/knowhere-rs/benchmark_results/
```

---

## Task 1: 诊断 — 观测 Milvus 实际送给 IVF-SQ8 的 nq 值

**Purpose:** 在实施任何优化前，确认 Milvus 是否批处理 IVF-SQ8 查询（nq>1），以选择正确的优化路径。

### Step 1: 启动带 TRACE_SEARCH 的 Milvus

首先确认 Milvus 以 RS 模式运行（当前状态应该是，上轮已恢复）：

```bash
ssh hannsdb-x86 'pgrep -fa "milvus run standalone" | grep -v grep'
```

若不在运行，重启：
```bash
ssh hannsdb-x86 'bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_rs.log 2>&1 &'
ssh hannsdb-x86 'sleep 15 && /data/work/VectorDBBench/.venv/bin/python3 -c "from pymilvus import connections; connections.connect(host=\"127.0.0.1\", port=\"19530\"); print(\"OK\")"'
```

由于 `KNOWHERE_RS_TRACE_SEARCH` 是 env var，直接在 Milvus 进程中设置。停止 Milvus，用 TRACE_SEARCH 重启：

```bash
ssh hannsdb-x86 'pkill -f "milvus run standalone"; sleep 5'
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src && KNOWHERE_RS_TRACE_SEARCH=1 nohup bash scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_trace.log 2>&1 &'
ssh hannsdb-x86 'sleep 15 && /data/work/VectorDBBench/.venv/bin/python3 -c "from pymilvus import connections; connections.connect(host=\"127.0.0.1\", port=\"19530\"); print(\"OK\")"'
```

**备注**: 若 start_standalone_remote.sh 覆盖 env var，需要直接启动 Milvus 二进制：
```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ/milvus-src && KNOWHERE_RS_TRACE_SEARCH=1 ETCD_USE_EMBED=true COMMON_STORAGETYPE=local LD_LIBRARY_PATH=/data/work/milvus-rs-integ/milvus-src/cmake-build/lib:/data/work/milvus-rs-integ/openblas/lib:$LD_LIBRARY_PATH nohup ./bin/milvus run standalone > /tmp/milvus_trace.log 2>&1 &'
ssh hannsdb-x86 'sleep 20 && tail -5 /tmp/milvus_trace.log'
```

### Step 2: 运行短版 benchmark（5s per point，仅 c=20 和 c=80）

```bash
ssh hannsdb-x86 'cat > /tmp/ivfsq8_diag_bench.py' << 'PYEOF'
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

COL = "ivfsq8_diag"
if utility.has_collection(COL):
    utility.drop_collection(COL)
col = Collection(COL, schema)
for i in range(0, N, 5000):
    col.insert([list(range(i, min(i+5000, N))), vecs[i:min(i+5000,N)].tolist()])
col.flush()
col.create_index("v", {"index_type": "IVF_SQ8", "metric_type": "IP", "params": {"nlist": NLIST}})
utility.wait_for_index_building_complete(COL)
col.load(); time.sleep(3)

np.random.seed(99)
all_queries = np.random.randn(500, DIM).astype(np.float32)
all_queries /= np.linalg.norm(all_queries, axis=1, keepdims=True)

def run_concurrent(concurrency, duration_secs=8):
    count, stop, errors = [0], [False], [0]
    def worker(qidx):
        while not stop[0]:
            try:
                col.search([all_queries[qidx % 500].tolist()], "v",
                    {"metric_type": "IP", "params": {"nprobe": 32}}, limit=10)
                count[0] += 1; qidx += concurrency
            except: errors[0] += 1
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(concurrency)]
    t0 = time.time()
    for t in threads: t.start()
    time.sleep(duration_secs)
    stop[0] = True
    for t in threads: t.join(timeout=5)
    return count[0] / (time.time() - t0), errors[0]

for c in [1, 20, 80]:
    qps, _ = run_concurrent(c)
    print(f"c={c}: {qps:.1f} QPS", flush=True)

utility.drop_collection(COL)
print("DONE")
PYEOF
ssh hannsdb-x86 'timeout 120 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_diag_bench.py 2>&1'
```

### Step 3: 分析 TRACE_SEARCH 日志中的 nq 分布

```bash
ssh hannsdb-x86 'grep "TRACE_SEARCH" /tmp/milvus_trace.log | grep -v "nq=0" | awk -F"nq=" "{print \$2}" | awk "{print \$1}" | sort | uniq -c | sort -rn | head -20'
```

Expected outcomes:
- **若 nq 通常 >1（大多数是 20-80）**: Milvus 在批处理 → 加 rayon 并行会有效 → 继续 Task 2A
- **若 nq 始终 =1**: Milvus 逐条分发 → rayon 并行无效 → 继续 Task 2B（shim 层合批）

### Step 4: 记录诊断结论

将观测到的 nq 分布和 QPS 写入诊断记录（口头汇报即可，勿写文件）。

---

## Task 2A: 若 nq>1 — 在 ivf.rs 加 IVF_NQ_POOL 并行

**触发条件**: Task 1 诊断显示 nq 经常 > NQ_PARALLEL_THRESHOLD(4)

**Files:**
- Modify: `src/faiss/ivf.rs` — 在 search() 加 rayon 并行
- Modify: `src/ffi.rs` — 若需共享 SendPtr 类型（已在 hnsw.rs 定义，可能需要 pub）

### Step 1: 在 ivf.rs 顶部添加 IVF_NQ_POOL 静态变量

在 `src/faiss/ivf.rs` 顶部（use 语句后）添加：

```rust
use once_cell::sync::Lazy;
use rayon;

const CGO_EXECUTOR_SLOTS: usize = 32;
const IVF_NQ_PARALLEL_THRESHOLD: usize = 4;

/// Copy+Send+Sync wrapper for raw pointer used in parallel nq writes.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    fn get(self) -> *mut T { self.0 }
}

static IVF_NQ_POOL: Lazy<rayon::ThreadPool> = Lazy::new(|| {
    let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
    let pool_threads = hw.saturating_sub(CGO_EXECUTOR_SLOTS).max(4);
    rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .thread_name(|i| format!("ivf-nq-{i}"))
        .build()
        .expect("failed to build IVF_NQ_POOL")
});
```

### Step 2: 修改 IvfSq8Index (IvfIndex) 的 search() 函数

在 `ivf.rs` 的 `fn search()` 实现（trait impl 中，处理 nq 的那个），将顺序 for 循环改为 rayon 并行：

当前代码（大约第 352-373 行）：
```rust
fn search(&self, query: &Dataset, top_k: usize) -> Result<IndexSearchResult, IndexError> {
    let q = query.vectors();
    if !q.len().is_multiple_of(self.dim) { return Err(IndexError::DimMismatch); }
    let nq = q.len() / self.dim;
    let mut ids = vec![-1; nq * top_k];
    let mut dists = vec![f32::MAX; nq * top_k];
    for qi in 0..nq {
        let query_vec = &q[qi * self.dim..(qi + 1) * self.dim];
        let hits = self.search_rows(query_vec, top_k, None);
        for (rank, (row, dist)) in hits.into_iter().enumerate() {
            ids[qi * top_k + rank] = self.ids.get(row).copied().unwrap_or(row as i64);
            dists[qi * top_k + rank] = dist;
        }
    }
    Ok(IndexSearchResult::new(ids, dists, 0.0))
}
```

改为（parallel path for nq >= threshold）：
```rust
fn search(&self, query: &Dataset, top_k: usize) -> Result<IndexSearchResult, IndexError> {
    let q = query.vectors();
    if !q.len().is_multiple_of(self.dim) { return Err(IndexError::DimMismatch); }
    let nq = q.len() / self.dim;
    let mut ids = vec![-1i64; nq * top_k];
    let mut dists = vec![f32::MAX; nq * top_k];

    if nq >= IVF_NQ_PARALLEL_THRESHOLD {
        // Pre-allocate flat output, parallel write via disjoint slices.
        let ids_ptr = SendPtr(ids.as_mut_ptr());
        let dists_ptr = SendPtr(dists.as_mut_ptr());
        let ids_ref = &ids_ptr;
        let dists_ref = &dists_ptr;
        IVF_NQ_POOL.install(|| {
            rayon::scope_fifo(|s| {
                for qi in 0..nq {
                    let ids_p = *ids_ref;
                    let dists_p = *dists_ref;
                    s.spawn_fifo(move |_| {
                        let query_vec = &q[qi * self.dim..(qi + 1) * self.dim];
                        let hits = self.search_rows(query_vec, top_k, None);
                        // SAFETY: each qi writes to disjoint slice [qi*top_k..(qi+1)*top_k]
                        unsafe {
                            let ids_slice = std::slice::from_raw_parts_mut(ids_p.get().add(qi * top_k), top_k);
                            let dists_slice = std::slice::from_raw_parts_mut(dists_p.get().add(qi * top_k), top_k);
                            for (rank, (row, dist)) in hits.into_iter().take(top_k).enumerate() {
                                ids_slice[rank] = self.ids.get(row).copied().unwrap_or(row as i64);
                                dists_slice[rank] = dist;
                            }
                        }
                    });
                }
            });
        });
    } else {
        for qi in 0..nq {
            let query_vec = &q[qi * self.dim..(qi + 1) * self.dim];
            let hits = self.search_rows(query_vec, top_k, None);
            for (rank, (row, dist)) in hits.into_iter().enumerate() {
                ids[qi * top_k + rank] = self.ids.get(row).copied().unwrap_or(row as i64);
                dists[qi * top_k + rank] = dist;
            }
        }
    }
    Ok(IndexSearchResult::new(ids, dists, 0.0))
}
```

**注意**: `IvfSq8Index` 是 `IvfIndex` 的 newtype wrapper（或 type alias），找到实际 search trait impl 位置并修改它。同样需要修改 `IvfFlatIndex` 的 search（若结构相同）。

### Step 3: 本地编译验证

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build 2>&1 | grep "^error"
```

预期：无 error。

### Step 4: 本地功能测试（快速 10K 测试）

```bash
cargo test ivf -- --nocapture 2>&1 | grep -E "PASSED|FAILED|error|ok"
```

### Step 5: rsync 到 x86 并编译

```bash
rsync -av --include="src/faiss/ivf.rs" --include="src/" --exclude="*" /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss/ivf.rs knowhere-x86-hk-proxy:/data/work/knowhere-rs-src/src/faiss/ivf.rs
ssh knowhere-x86-hk-proxy "cd /data/work/knowhere-rs-src && CARGO_TARGET_DIR=/data/work/knowhere-rs-target ~/.cargo/bin/cargo build --release 2>&1 | tail -5"
```

然后 rsync 到 hannsdb-x86 Milvus 集成目录（knowhere-rs 子模块）：

```bash
rsync -av /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/ hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/src/
ssh hannsdb-x86 "cd /data/work/milvus-rs-integ && cargo build --release --manifest-path knowhere-rs/Cargo.toml 2>&1 | tail -5"
```

### Step 6: 重启 Milvus（不带 TRACE_SEARCH），运行 full 15s benchmark

```bash
ssh hannsdb-x86 'pkill -f "milvus run standalone"; sleep 5'
ssh hannsdb-x86 'nohup bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_rs.log 2>&1 &'
ssh hannsdb-x86 'sleep 15 && /data/work/VectorDBBench/.venv/bin/python3 -c "from pymilvus import connections; connections.connect(host=\"127.0.0.1\", port=\"19530\"); print(\"OK\")"'
ssh hannsdb-x86 'timeout 900 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_concurrent_bench.py 2>&1 | tee /tmp/ivfsq8_nqpool_results.txt'
```

### Step 7: 收集结果

```bash
ssh hannsdb-x86 'cat /tmp/ivfsq8_nqpool_results.txt'
```

---

## Task 2B: 若 nq=1 — 诊断 Milvus dispatch 瓶颈（另开计划）

**触发条件**: Task 1 诊断显示 nq 始终 =1

若 nq 始终 1，加 rayon 并行无效。真正瓶颈是 Milvus 给 IVF-SQ8 的并发度上限（~4 线程）。
需要研究：
- Milvus CGO executor 对 IVF vs HNSW 是否有不同调度策略
- 是否可以在 C++ shim 层实现 query coalescing（累积 N 个 pending query 再一次性调用 RS）

这需要另开计划，不在本 Task 范围内。

---

## Task 3: 结果分析 + wiki 更新

### Step 1: 对比 RS nqpool vs RS base vs native

| Concurrency | nprobe=8 RS base | nprobe=8 RS+nqpool | nprobe=8 Native |
|-------------|-----------------|---------------------|-----------------|
| c=1  | 40.3 | ? | 38.7 |
| c=20 | 139.0 | ? | 138.8 |
| c=80 | 139.5 | ? | 141.1 |

期望：若 nq>1，c=80 QPS 从 140→500+ QPS。

### Step 2: 更新 wiki/benchmarks/authority-numbers.md

在 IVF-SQ8 Milvus 并发 QPS 章节添加 nqpool 优化结果。

### Step 3: 更新 wiki/log.md

添加 IVF-SQ8 nqpool 优化结果条目。

---

## Success Criteria

1. nq 分布诊断完成（nq>1 确认或否定）
2. 若 nq>1：IVF_NQ_POOL 实现并验证正确（本地测试通过）
3. 若 nq>1：Milvus benchmark 显示 c=80 QPS 显著提升（目标 >400 QPS）
4. Wiki 更新完成

## Anti-patterns to Avoid

- 不要跳过诊断直接实现 NQ_POOL（若 nq=1，实现白费）
- 不要在不理解根因的情况下随意修改 CGO_EXECUTOR_SLOTS 常量
- 若 nq>1 但 QPS 提升不显著：检查 IVF_NQ_POOL 线程数是否合理
