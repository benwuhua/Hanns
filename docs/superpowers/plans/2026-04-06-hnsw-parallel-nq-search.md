# HNSW Parallel nq Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `search_with_bitset_ref` 中的串行 nq 循环改为 rayon 并行，消除 RS HNSW 在高并发场景下的 QPS 瓶颈。

**Architecture:** 当前 `search_with_bitset_ref` 用串行 `for q_idx in 0..n_queries` 逐个处理查询，而 native HNSW 用线程池 fan-out（`search_pool_->push()`）并行处理所有 nq queries。rayon `into_par_iter()` 已在文件顶部 import（line 13），每个查询的 scratch buffer 均用 TLS（`HNSW_SEARCH_SCRATCH_TLS`），并发安全。同时清理已完成诊断任务的遗留 statics（`HNSW_CONCURRENT_SEARCHES` 等）。

**Tech Stack:** Rust, rayon (already in Cargo.toml), thread-local storage

**Background:**
- 文件: `src/faiss/hnsw.rs`
- 根因: line 6759 `for q_idx in 0..n_queries` 串行，native 并行 fan-out
- rayon: `use rayon::prelude::*` 已在 line 13
- TLS 安全: `HNSW_SEARCH_SCRATCH_TLS`, `HNSW_LATENCY_SAMPLES`, `HNSW_COSINE_QUERY_NORM_TLS`, `HNSW_SQ_PRECOMPUTED_TLS` 全为 thread-local，rayon worker 各有独立 TLS
- `BitsetRef<'a>` 只含 `&'a [u64]` + `usize`，自动 `Sync + Send`
- `HnswIndex` struct 无 `Cell`/`RefCell` 字段（均在 TLS），自动 `Sync`

**Deployment:**
- 本地: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- 远端源码: `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs`
- Build cache: `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs-target`
- VectorDBBench: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
- Milvus log: `/data/work/milvus-rs-integ/milvus-src/standalone.log`

---

## File Structure

- Modify: `src/faiss/hnsw.rs` — 仅改此文件

---

## Task 1：清理遗留并发诊断 statics

诊断目的已完成（peak_concurrent 数据已收集），清理遗留的 3 个 statics 和 `ConcurrencyGuard`，以及 `search_with_bitset_ref` 中使用它们的代码块。

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: 移除 3 个 statics 和 ConcurrencyGuard（lines 88-97 附近）**

找到以下代码块并**完整删除**：

```rust
static HNSW_CONCURRENT_SEARCHES: AtomicI32 = AtomicI32::new(0);
static HNSW_SEARCH_CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
static HNSW_PEAK_CONCURRENT: AtomicI32 = AtomicI32::new(0);

struct ConcurrencyGuard;
impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        HNSW_CONCURRENT_SEARCHES.fetch_sub(1, Ordering::Relaxed);
    }
}
```

- [ ] **Step 2: 移除 `search_with_bitset_ref` 中的并发诊断代码块（lines 6726-6748 附近）**

在 `search_with_bitset_ref` 函数中，找到以下诊断代码块并**完整删除**（从 `// Concurrency diagnostic` 到 `let _guard = ConcurrencyGuard;`，共约 23 行）：

```rust
        // Concurrency diagnostic
        let concurrent_now = HNSW_CONCURRENT_SEARCHES.fetch_add(1, Ordering::Relaxed) + 1;
        let call_num = HNSW_SEARCH_CALLS_TOTAL.fetch_add(1, Ordering::Relaxed) + 1;
        // Update peak
        let mut peak = HNSW_PEAK_CONCURRENT.load(Ordering::Relaxed);
        while concurrent_now > peak {
            match HNSW_PEAK_CONCURRENT.compare_exchange_weak(
                peak, concurrent_now, Ordering::Relaxed, Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => peak = x,
            }
        }
        // Every 1000 calls, dump stats to stderr
        if call_num % 1000 == 0 {
            eprintln!(
                "[HNSW_CONC] calls={} peak_concurrent={} current_at_entry={}",
                call_num,
                HNSW_PEAK_CONCURRENT.load(Ordering::Relaxed),
                concurrent_now
            );
        }
        let _guard = ConcurrencyGuard;
```

- [ ] **Step 3: 检查并移除不再使用的 import**

检查文件顶部 `use std::sync::atomic` import 行，确认 `AtomicI32` 是否还有其他使用（用 grep 确认）：

```bash
grep -n "AtomicI32" src/faiss/hnsw.rs
```

如果只剩上面已删的 3 处引用（statics 定义 + ConcurrencyGuard），则从 import 行移除 `AtomicI32`。如果还有其他使用，保留。

同样检查 `AtomicU64`（只用于 `HNSW_SEARCH_CALLS_TOTAL`）：

```bash
grep -n "AtomicU64" src/faiss/hnsw.rs
```

如果只剩 0 处，从 import 移除。

- [ ] **Step 4: 编译验证**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

Expected: `Build OK`

- [ ] **Step 5: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "cleanup(hnsw): remove stale concurrency diagnostic statics and guard"
```

---

## Task 2：并行化 nq 循环

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: 替换 `search_with_bitset_ref` 中的串行 nq 循环**

在 `search_with_bitset_ref` 函数中，找到以下代码（Task 1 完成后 statics 已移除，约在原 line 6756 位置）：

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

替换为：

```rust
        // Parallel nq search: each query is independent; scratch buffers are TLS-safe.
        let per_query: Vec<Vec<(i64, f32)>> = (0..n_queries)
            .into_par_iter()
            .map(|q_idx| {
                let q_start = q_idx * self.dim;
                let query_vec = &query[q_start..q_start + self.dim];
                self.search_single_with_bitset_ref(query_vec, ef, k, bitset)
            })
            .collect();

        let mut all_ids = Vec::with_capacity(n_queries * k);
        let mut all_dists = Vec::with_capacity(n_queries * k);
        for pairs in per_query {
            for (id, dist) in pairs.into_iter().take(k) {
                all_ids.push(id);
                all_dists.push(dist);
            }
        }
```

注意：`rayon::prelude::*` 已在文件 line 13 import，`into_par_iter()` 可直接使用。

- [ ] **Step 2: 编译验证**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

Expected: `Build OK`

如果出现编译错误，常见原因及修复：
- `BitsetRef is not Send/Sync`: 不应出现（`BitsetRef<'a>` 含 `&'a [u64]` + `usize`，自动 Sync）
- `cannot borrow ... as immutable because it is also borrowed as mutable`: 不适用（`self` 是 `&self`，无可变借用）
- lifetime 错误: 检查 `bitset` 参数生命周期，可能需要在 closure 加 `move` 或调整 lifetime bound

- [ ] **Step 3: 运行相关测试**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo test hnsw 2>&1 | tail -15
```

Expected: 所有 hnsw 相关测试通过（no FAILED）

- [ ] **Step 4: Commit**

```bash
git add src/faiss/hnsw.rs
git commit -m "perf(hnsw): parallelize nq loop in search_with_bitset_ref using rayon

Replace serial for-loop over n_queries with rayon into_par_iter().
Each query is independent; scratch buffers use TLS (HNSW_SEARCH_SCRATCH_TLS)
so rayon worker threads are safe. Mirrors native HNSW search_pool_->push()
fan-out pattern.

Root cause: [HNSW_WINDOW] diag showed duration_avg=27.1ms at concurrency=2
vs expected ~6ms (2 queries × ~2.87ms each), confirming serial nq processing.
Native uses ~80-thread pool, RS was using 1 thread for all nq queries."
```

---

## Task 3：部署到 hannsdb-x86，验证 QPS 改善（Codex 负责）

> **注意**: 这个 task 由 Codex 通过 tmux 在 hannsdb-x86 上执行。

**Files:** 无代码改动，仅部署+验证

- [ ] **Step 1: rsync 本地 src/ 到远端**

```bash
rsync -av /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/ \
    ryan@189.1.218.159:/data/work/milvus-rs-integ/knowhere-rs/src/
```

- [ ] **Step 2: 远端 cargo build --release**

```bash
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ/knowhere-rs && \
    CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
    ~/.cargo/bin/cargo build --release 2>&1 | tail -5"
```

Expected: `Finished release [optimized]`

- [ ] **Step 3: 确认 Milvus 已停止，然后重启**

```bash
ssh ryan@189.1.218.159 "pkill -f 'milvus run standalone' || true; sleep 3"
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ/milvus-src && \
    nohup ./bin/milvus run standalone > standalone.log 2>&1 &"
sleep 10
ssh ryan@189.1.218.159 "grep -c 'Milvus Proxy listen' /data/work/milvus-rs-integ/milvus-src/standalone.log || echo 'not ready yet'"
```

如果 not ready，再等 10 秒后重试 grep。

- [ ] **Step 4: 运行 VectorDBBench（默认配置，记录 QPS）**

```bash
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ && \
    python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py \
    2>/tmp/rs_parallel_nq.log; echo EXIT_CODE=$?"
```

等待完成（约 20-30 分钟）。

- [ ] **Step 5: 从日志中提取 QPS 数字**

```bash
ssh ryan@189.1.218.159 "grep -i 'qps\|query.*per\|throughput' /tmp/rs_parallel_nq.log | tail -10"
```

同时检查 Milvus log 确认没有 panic：

```bash
ssh ryan@189.1.218.159 "grep -i 'panic\|fatal\|SIGABRT' \
    /data/work/milvus-rs-integ/milvus-src/standalone.log | tail -5"
```

- [ ] **Step 6: 写入结论**

把结果写入 `/tmp/codex_status.txt`：

```
DONE: QPS=<N> (vs R4 baseline=349, native=350/848), recall=<R>
```

如果 QPS 明显高于 349（预期 ≥2×），说明并行化有效。如果 QPS 不变或更低，说明有其他瓶颈（如 Milvus query coordinator 或 rayon 线程竞争）。
