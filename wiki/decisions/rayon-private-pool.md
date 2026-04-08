# 为什么不能用 rayon 全局池

**结论**：在 Milvus CGO executor 线程中调用 rayon 全局池（`into_par_iter()`）会导致线程过度订阅，QPS 骤降。必须使用**私有 ThreadPool + `install()` 模式**。

---

## 失败案例：R6 rayon nq 并行（2026-04-06）

**尝试**：在 HNSW 的 `search_batch_with_bitset` 中把 `for q_idx in 0..n_queries` 改为 `into_par_iter()`。

**结果**：QPS 349→170 (−51%)，Optimize 336.9s→4464s (13×)，Load 673.7s→4618s (7×)。

**Commit**：`89e99f3`（已 revert）

---

## 根因分析

### Milvus CGO executor 线程模型

```
Milvus QueryNode
  └─ CGO_EXECUTOR_SLOTS = 32 个 goroutine
       └─ 每个 goroutine → C.AsyncSearch()
            └─ 在 CGO executor 线程上运行
```

32 个 CGO executor 线程同时调用 RS 的 search 函数。

### rayon 全局池的问题

```
into_par_iter() 调用者线程 JOIN 全局 rayon pool
→ 32 CGO 线程各自招募 N 个 rayon worker 线程
→ 总线程数 = 32 × (1 + rayon_pool_size)
→ 严重过度订阅（hw=16 cores，但有 32×N 个线程竞争）
→ 频繁 context switch → 性能崩溃
```

### Little's Law 分析

过度订阅后：
- 每个 query 的有效延迟大幅增加（线程等待调度）
- 吞吐量反而下降（调度开销超过并行收益）

---

## 正确方案：HNSW_NQ_POOL（R7）

```rust
static HNSW_NQ_POOL: once_cell::Lazy<rayon::ThreadPool> = once_cell::Lazy::new(|| {
    let hw = num_cpus::get();
    let pool_threads = (hw.saturating_sub(32)).max(4);  // 留余量给 CGO executor
    rayon::ThreadPoolBuilder::new()
        .num_threads(pool_threads)
        .build()
        .expect("failed to build HNSW NQ pool")
});
```

### 使用方式

```rust
HNSW_NQ_POOL.install(|| {
    queries.par_chunks(dim).enumerate().for_each(|(q_idx, q)| {
        // 每个 query 的搜索
    })
});
```

**`install()` 的关键**：调用者线程作为 work-stealer 加入**私有池**，而不是全局池。私有池线程数固定，不会无限增长。

### 参数（hannsdb-x86）

- hw = 16 cores
- CGO_EXECUTOR_SLOTS = 32
- pool_threads = (16 - 32).max(4) = **4**（min floor）

---

## 效果

| Round | 方案 | c=20 QPS | c=80 QPS |
|-------|------|---------|---------|
| R4 | 无并行（serial nq） | — | 349 |
| R6 | rayon 全局池 | — | 170 ← 失败 |
| R7 | **HNSW_NQ_POOL** | **548** | **540** |
| R8 | R7 + alloc-reduction | **1051** | **1042** |

---

## 可复用规律

1. **在 CGO/C FFI 调用的线程中，永远不要用 rayon 全局 `into_par_iter()`**。
2. **私有池大小 = `(hw - cgo_slots).max(floor)`**：给 CGO executor 留够 CPU 时间。
3. **`install()` 而非 `spawn()`**：调用者线程参与 work-stealing，减少线程切换。
4. **`scope_fifo()`** 而非 `scope()`：保证 FIFO 顺序，防止 head-of-line blocking。

---

## 相关页面

- [[concepts/hnsw]] — HNSW Milvus 集成优化历史
- [[concepts/milvus-integration]] — CGO executor 线程模型
- [[decisions/optimization-log]] — 完整优化时间线
