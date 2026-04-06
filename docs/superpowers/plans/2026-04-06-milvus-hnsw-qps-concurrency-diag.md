# Milvus HNSW QPS 并发诊断计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 确认 RS HNSW QPS 并发瓶颈的根因：是线程池饱和（32 线程被 1 query 占满）还是外部序列化（Milvus 层锁/队列）。

**Architecture:** 在 `ffi.rs` 的 `knowhere_search_with_bitset` HNSW 分支加 `[HNSW_WINDOW]` 仪器，追踪"并发搜索窗口"。当 in_flight 从 0→1 时记录窗口开始，从 N→0 时记录 peak 并发数和窗口时长。用 concurrency=2 跑 VectorDBBench：如果 peak_concurrent > 32 → query 可以重叠（线程池不是瓶颈）；如果始终 ≤ 32 → 一次只处理 1 query（线程池饱和或外部序列化）。

**Tech Stack:** Rust `std::sync::atomic`，`std::time::SystemTime`，hannsdb-x86 VectorDBBench

**Background:**
- RS R4: QPS=349，native: QPS=350（低并发下 parity）
- native 在高并发时能扩展到 ~848 QPS，RS 始终停在 349
- Little's Law: 349 QPS × 2.87ms = 1.0 → 任意时刻只有 1 个 query 在处理
- `HnswRustNode::Search()`、`Index<T>::Search()` 均无 mutex
- peak_concurrent=32 = Cohere-1M segment 数 → 1 query 用 32 线程
- **待确认**：多个 query 能否在 RS FFI 层重叠？

**Deployment:**
- 本地: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- 远端源码: `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs`
- Build cache: `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs-target`
- VectorDBBench: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
- Milvus log（含 RS stderr）: `/data/work/milvus-rs-integ/milvus-src/standalone.log` 或 `standalone-stage1.log`

---

## File Structure

- Modify: `src/ffi.rs` — `knowhere_search_with_bitset` 函数（添加 `[HNSW_WINDOW]` 仪器）

---

## Task 1：添加 [HNSW_WINDOW] 搜索窗口诊断

**Purpose:**
- `peak_concurrent` > 32 → 多个 query 能在 FFI 层重叠 → 线程池不是根因
- `peak_concurrent` 始终 ≤ 32 → FFI 层每次只有 1 query → 线程池饱和或外部序列化

**Files:**
- Modify: `src/ffi.rs`

- [ ] **Step 1: 添加 import 和全局变量**

在 `src/ffi.rs` 文件顶部（`const FFI_FORCE_SERIAL...` 之前），添加：

```rust
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering as AtomicOrdering};

// [HNSW_WINDOW] 搜索并发窗口诊断
static SEARCH_IN_FLIGHT: AtomicI32 = AtomicI32::new(0);
static SEARCH_PEAK_CONCURRENT: AtomicI32 = AtomicI32::new(0);
static SEARCH_WINDOW_COUNT: AtomicU64 = AtomicU64::new(0);
static SEARCH_WINDOW_START_NS: AtomicU64 = AtomicU64::new(0);
```

注意：如果 `AtomicI32`/`AtomicU64` 已在文件中 import，只添加缺少的类型。

- [ ] **Step 2: 在 `knowhere_search_with_bitset` 的 HNSW 分支插入仪器**

找到 `src/ffi.rs` 中 `knowhere_search_with_bitset` 函数（约第 2507 行），在 HNSW 分支：

```rust
} else if let Some(ref idx) = index.hnsw {
    match idx.search_with_bitset_ref(query_slice, &req, &bitset_ref) {
```

改为（在搜索调用前后插入窗口追踪）：

```rust
} else if let Some(ref idx) = index.hnsw {
    // [HNSW_WINDOW] entry: track concurrent searches across all threads
    let prev_if = SEARCH_IN_FLIGHT.fetch_add(1, AtomicOrdering::Relaxed);
    let cur_if = prev_if + 1;
    // Record window start when first search enters
    if prev_if == 0 {
        let now_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        SEARCH_WINDOW_START_NS.store(now_ns, AtomicOrdering::Relaxed);
    }
    // CAS loop to update peak
    {
        let mut peak = SEARCH_PEAK_CONCURRENT.load(AtomicOrdering::Relaxed);
        loop {
            if cur_if <= peak {
                break;
            }
            match SEARCH_PEAK_CONCURRENT.compare_exchange_weak(
                peak,
                cur_if,
                AtomicOrdering::Relaxed,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => break,
                Err(p) => peak = p,
            }
        }
    }

    let hnsw_result = idx.search_with_bitset_ref(query_slice, &req, &bitset_ref);

    // [HNSW_WINDOW] exit: log window stats when last search completes
    let remaining_if = SEARCH_IN_FLIGHT.fetch_sub(1, AtomicOrdering::Relaxed) - 1;
    if remaining_if == 0 {
        let window_n = SEARCH_WINDOW_COUNT.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        let start_ns = SEARCH_WINDOW_START_NS.load(AtomicOrdering::Relaxed);
        let end_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let duration_ms = (end_ns.saturating_sub(start_ns)) as f64 / 1_000_000.0;
        let peak = SEARCH_PEAK_CONCURRENT.swap(0, AtomicOrdering::Relaxed);
        if window_n <= 20 || window_n % 200 == 0 {
            eprintln!(
                "[HNSW_WINDOW] window={} peak_concurrent={} duration_ms={:.1}",
                window_n, peak, duration_ms
            );
        }
    }

    match hnsw_result {
```

**注意**：原来的 `match idx.search_with_bitset_ref(...)` 拆成两步：先 `let hnsw_result = ...`，再 `match hnsw_result {`。其余 match 分支代码不变。

- [ ] **Step 3: 编译验证**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

Expected: `Build OK`

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "diag(ffi): add [HNSW_WINDOW] concurrent search window diagnostic"
```

---

## Task 2：部署到 hannsdb-x86，用 concurrency=2 运行诊断（Codex 负责）

> **注意**: 这个 task 由 Codex 通过 tmux 在 hannsdb-x86 上执行。

- [ ] **Step 1: rsync 本地 src/ 到远端**

```bash
rsync -av /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/ \
    ryan@189.1.218.159:/data/work/milvus-rs-integ/knowhere-rs/src/
```

- [ ] **Step 2: 远端 cargo build --release**

```bash
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ/knowhere-rs && \
    CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
    ~/.cargo/bin/cargo build --release 2>&1 | tail -3"
```

Expected: `Finished release [optimized]`

- [ ] **Step 3: 确认 VectorDBBench 脚本支持 concurrency 参数**

```bash
ssh ryan@189.1.218.159 "head -50 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py | grep -i concurr"
```

如果脚本有 concurrency 参数（如 `--concurrency 2`），用参数。如果是硬编码，临时修改脚本或复制一份改 concurrency=2。

- [ ] **Step 4: 运行 VectorDBBench（concurrency=2）**

```bash
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ && \
    python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py \
    2>/tmp/rs_window_diag.log; echo EXIT_CODE=$?"
```

等待完成（约 20-30 分钟）。

- [ ] **Step 5: 提取 [HNSW_WINDOW] 日志行**

从 Milvus standalone log 中提取（之前发现诊断输出在 Milvus log，不在客户端 stderr）：

```bash
ssh ryan@189.1.218.159 "grep '\[HNSW_WINDOW\]' \
    /data/work/milvus-rs-integ/milvus-src/standalone*.log 2>/dev/null | head -30"
```

如果为空，尝试从 `/tmp/rs_window_diag.log` 查找：

```bash
ssh ryan@189.1.218.159 "grep '\[HNSW_WINDOW\]' /tmp/rs_window_diag.log | head -30"
```

- [ ] **Step 6: 分析并写入结论**

分析 `peak_concurrent` 字段：

| 情况 | 含义 | 下一步 |
|------|------|--------|
| peak_concurrent > 32（如 64） | query 在 RS FFI 层可重叠 | 线程池不是根因；向上追 Milvus 层序列化 |
| peak_concurrent 始终 = 32 | 一次只有 1 query | 线程池饱和（32 线程）或外部序列化 |
| peak_concurrent < 32（如 16） | 每个 query 用的线程更少 | 说明 concurrency=2 下 query 确实在重叠（16+16=32） |

同时记录：
- `duration_ms`：平均窗口时长（应约等于单 query 搜索时间，即 ~2.87ms）
- 窗口之间有无明显间隔（看相邻 window 时间戳）

把结论写入 `/tmp/codex_status.txt`：
```
DONE: peak_concurrent=<N>, duration_ms=<X>, 结论=<线程池饱和/query重叠/其他>
```

---

## Task 3：移除诊断代码

诊断目的完成后，恢复干净的 `knowhere_search_with_bitset`。

**Files:**
- Modify: `src/ffi.rs`

- [ ] **Step 1: 移除 4 个全局 static（SEARCH_IN_FLIGHT/PEAK/WINDOW_COUNT/WINDOW_START_NS）**

- [ ] **Step 2: 恢复 HNSW 分支**

将 Task 1 修改的 HNSW 分支恢复为原始形式：

```rust
} else if let Some(ref idx) = index.hnsw {
    match idx.search_with_bitset_ref(query_slice, &req, &bitset_ref) {
        Ok(result) => {
            // ... (原有代码不变)
        }
        Err(_) => std::ptr::null_mut(),
    }
```

- [ ] **Step 3: 移除不再使用的 import（如果 AtomicI32/AtomicU64 仅供诊断用）**

- [ ] **Step 4: 编译验证**

```bash
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs
git commit -m "cleanup(ffi): remove [HNSW_WINDOW] diagnostic instrumentation"
```

---

## 诊断结论解读

### 如果 peak_concurrent > 32（query 可重叠）

说明 RS FFI 层没有序列化，Milvus 可以让多个 query 并发进入 knowhere_search_with_bitset。
下一步：找 Milvus 层的序列化点（Go goroutine/channel 设计、search coordinator 逻辑）。

### 如果 peak_concurrent 始终 = 32（一次只有 1 query）

**子假说 A：线程池大小 = 32**
- Milvus QueryNode 搜索线程池只有 32 个线程
- 1 query × 32 segments 用光所有线程
- Fix：增大线程池（找 QueryNode 的 `search_pool` 配置项，改为 80 或更大）

**子假说 B：外部序列化**
- 线程池可能更大，但某处有互斥让 query 串行进入 RS 路径
- Fix：找该互斥并改为 shared_lock 或无锁

区分 A vs B：检查 duration_ms 与 inter-window gap 的比例。如果 gap ≈ 0 → 子假说 A（线程立即被复用，只是同时不够多）。如果 gap 明显 > 0 → 子假说 B（有东西让 query 等待）。
