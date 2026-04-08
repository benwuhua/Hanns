# IVF-SQ8 私有 Cluster Pool 优化计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 IVF-SQ8 cluster scanning 创建私有 rayon pool（镜像 HNSW R7），消除 80 个 CGO 线程打全局 pool 的过订问题，突破 135 QPS 天花板。

**Root Cause (confirmed):**
- IVF-SQ8 search 的 `#[cfg(feature = "parallel")]` 路径：每个 query 对 nprobe 个 cluster 做 `clusters.par_iter()` → 打全局 rayon pool
- 80 并发 CGO 线程 × nprobe=8 = 640 tasks 争抢 16 pool threads → 过订 40×
- 实测验证：nprobe=8 → 25.5ms，nprobe=32 → 61.3ms，差值/24 = 1.49ms/cluster（顺序模型而非并行）
- 同 HNSW R6 bug（rayon 全局池过订），R7 用私有池修复

**Architecture:** 在 `ivf_sq8.rs` 添加 `IVF_SQ8_CLUSTER_POOL`（静态私有 rayon pool），将 `par_iter()` 包在 `IVF_SQ8_CLUSTER_POOL.install(|| ...)` 中。CGO executor 线程通过 `install()` 临时加入私有池协作，避免全局池竞争。

**Tech Stack:** Rust (once_cell, rayon), src/faiss/ivf_sq8.rs

---

## Context

### 关键代码位置

`src/faiss/ivf_sq8.rs` 第 373-398 行：
```rust
#[cfg(feature = "parallel")]
let merged = {
    use rayon::prelude::*;
    let dim = self.dim;
    let partials: Vec<TopKAccumulator> = clusters
        .par_iter()   // ← 此处直接打全局 pool
        .map_init(...)
        .collect();
    ...
};
```

### HNSW_NQ_POOL 参照（hnsw.rs）

```rust
const CGO_EXECUTOR_SLOTS: usize = 32;
const NQ_PARALLEL_THRESHOLD: usize = 4;

static HNSW_NQ_POOL: once_cell::sync::Lazy<rayon::ThreadPool> =
    once_cell::sync::Lazy::new(|| {
        let hw = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
        let pool_threads = hw.saturating_sub(CGO_EXECUTOR_SLOTS).max(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .thread_name(|i| format!("hnsw-nq-{i}"))
            .build()
            .expect("failed to build HNSW_NQ_POOL")
    });

// 使用：
HNSW_NQ_POOL.install(|| {
    rayon::scope_fifo(|s| { ... });
});
```

### Key Paths

```
Source: /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss/ivf_sq8.rs
Remote x86 src: hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/src/faiss/ivf_sq8.rs
Cargo target: hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs-target/
Benchmark: /tmp/ivfsq8_concurrent_bench.py (full 15s, nlist=1024)
Diag benchmark: /tmp/ivfsq8_diag.py (8s, c=1/20/80)
```

---

## Task 1: 实现 IVF_SQ8_CLUSTER_POOL

**Files:** Modify `src/faiss/ivf_sq8.rs`

### Step 1: 在文件顶部 use 语句区添加 pool 静态变量

在 `ivf_sq8.rs` 的顶部 use 区（文件开头几十行，找到其他 use 语句后），添加：

```rust
/// 私有 rayon pool，用于 IVF-SQ8 cluster 并行 scanning。
/// 镜像 hnsw.rs 的 HNSW_NQ_POOL 模式：CGO executor 线程通过 install() 加入，
/// 避免全局 rayon pool 过订（80 并发 × nprobe tasks = 640 tasks 争 16 线程）。
#[cfg(feature = "parallel")]
const IVF_SQ8_CGO_EXECUTOR_SLOTS: usize = 32;

#[cfg(feature = "parallel")]
static IVF_SQ8_CLUSTER_POOL: once_cell::sync::Lazy<rayon::ThreadPool> =
    once_cell::sync::Lazy::new(|| {
        let hw = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(8);
        let pool_threads = hw.saturating_sub(IVF_SQ8_CGO_EXECUTOR_SLOTS).max(4);
        rayon::ThreadPoolBuilder::new()
            .num_threads(pool_threads)
            .thread_name(|i| format!("ivfsq8-cluster-{i}"))
            .build()
            .expect("failed to build IVF_SQ8_CLUSTER_POOL")
    });
```

### Step 2: 将 par_iter() 包进 IVF_SQ8_CLUSTER_POOL.install()

在 `src/faiss/ivf_sq8.rs` 第 373-398 行的 `#[cfg(feature = "parallel")]` 块，将：

```rust
let partials: Vec<TopKAccumulator> = clusters
    .par_iter()
    .map_init(
        || (vec![0.0f32; dim], vec![0i16; dim]),
        |(q_residual_buf, q_precomputed_buf), &cluster_id| {
            self.scan_cluster_with_buf(...)
        },
    )
    .collect();
```

改为：

```rust
let partials: Vec<TopKAccumulator> = IVF_SQ8_CLUSTER_POOL.install(|| {
    clusters
        .par_iter()
        .map_init(
            || (vec![0.0f32; dim], vec![0i16; dim]),
            |(q_residual_buf, q_precomputed_buf), &cluster_id| {
                self.scan_cluster_with_buf(...)
            },
        )
        .collect()
});
```

### Step 3: 确认 once_cell 已在 Cargo.toml 依赖中

```bash
grep "once_cell" /Users/ryan/.openclaw/workspace-builder/knowhere-rs/Cargo.toml
```

若无，需要添加（但 hnsw.rs 已用 once_cell，说明已有依赖）。

### Step 4: 本地编译验证

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build 2>&1 | grep "^error"
```

预期：无 error。

### Step 5: 本地功能测试

```bash
cargo test ivf_sq8 -- --nocapture 2>&1 | grep -E "PASSED|FAILED|error|ok|test " | head -20
```

确认 IVF-SQ8 相关测试通过。

---

## Task 2: rsync + x86 编译 + Milvus 重启

### Step 1: rsync ivf_sq8.rs 到 hannsdb-x86

```bash
rsync -av /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss/ivf_sq8.rs \
    hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs/src/faiss/ivf_sq8.rs
```

### Step 2: 在 hannsdb-x86 上编译 RS library

```bash
ssh hannsdb-x86 'cd /data/work/milvus-rs-integ && \
    CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
    cargo build --release --manifest-path knowhere-rs/Cargo.toml 2>&1 | tail -10'
```

预期：`Compiling knowhere_rs ... Finished release`

### Step 3: 重启 Milvus（保留数据）

```bash
ssh hannsdb-x86 'pkill -f "milvus run standalone"; sleep 5; echo stopped' 2>&1 || true
ssh hannsdb-x86 'pgrep -f "milvus run standalone" | wc -l'
```

若已停止（0），重启：
```bash
ssh hannsdb-x86 'RESET_RUNTIME_STATE=false nohup bash /data/work/milvus-rs-integ/milvus-src/scripts/knowhere-rs-shim/start_standalone_remote.sh > /tmp/milvus_clusterPool.log 2>&1 &'
ssh hannsdb-x86 'sleep 20 && /data/work/VectorDBBench/.venv/bin/python3 -c "from pymilvus import connections; connections.connect(host=\"127.0.0.1\", port=\"19530\"); print(\"Milvus OK\")"'
```

---

## Task 3: 诊断 benchmark + 分析

### Step 1: 运行快速诊断（8s per point）

```bash
ssh hannsdb-x86 'timeout 120 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_diag.py 2>&1'
```

**预期（若 pool 修复有效）**：
- c=1: ~50-70 QPS（install() 开销允许更高吞吐，或接近 66 QPS = 1/(H+1.49ms)）
- c=20: 200-400+ QPS
- c=80: 400-800+ QPS

**若 c=1 QPS 降低（<39）**: install() 开销过大，需要增加 CLUSTER_POOL_THRESHOLD（仅在 nprobe 超过阈值时才用 pool）。

**若 c=80 QPS 无改善（仍 ~135）**: rayon pool 不是瓶颈，需要其他方向。

### Step 2: 若诊断有效，运行完整 15s benchmark

```bash
ssh hannsdb-x86 'timeout 900 /data/work/VectorDBBench/.venv/bin/python3 /tmp/ivfsq8_concurrent_bench.py 2>&1 | tee /tmp/ivfsq8_clusterPool_results.txt'
ssh hannsdb-x86 'cat /tmp/ivfsq8_clusterPool_results.txt'
```

---

## Task 4: 提交 + 结果记录 + wiki 更新

### Step 1: git commit（本地）

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
git add src/faiss/ivf_sq8.rs
git commit -m "perf(ivf_sq8): IVF_SQ8_CLUSTER_POOL — private rayon pool for cluster scanning

Mirror HNSW R7 pattern: CGO executor threads join private pool via
install(), avoiding global pool oversubscription.

80 concurrent CGO threads × nprobe tasks were submitting to the global
rayon pool (640 tasks vs 16 threads = 40× oversubscription). Private
pool isolates IVF-SQ8 cluster work, allowing proper cooperation."
```

### Step 2: 保存结果到本地

保存到 `benchmark_results/ivf_sq8_milvus_cluster_pool_2026-04-08.md`

### Step 3: 更新 wiki

- `wiki/benchmarks/authority-numbers.md`：添加 IVF-SQ8 cluster pool 结果（Round X）
- `wiki/log.md`：新增条目

---

## Success Criteria

1. `IVF_SQ8_CLUSTER_POOL.install()` 代码正确编译
2. 本地 ivf_sq8 单元测试通过
3. Milvus benchmark 显示 c=80 QPS 显著提升（目标 >300 QPS，理想 >500 QPS）
4. 代码 commit
5. Wiki 更新

## Anti-patterns to Avoid

- 不要调整 pool 线程数（用默认 `(hw - 32).max(4)` 与 HNSW 相同）
- 不要删除 serial fallback（`#[cfg(not(feature = "parallel"))]` 保留）
- 若 c=1 QPS 明显下降：添加 `const IVF_SQ8_CLUSTER_PARALLEL_THRESHOLD: usize = 4;` 只在 nprobe >= threshold 时用 pool
