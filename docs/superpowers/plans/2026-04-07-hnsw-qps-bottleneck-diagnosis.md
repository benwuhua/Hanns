# HNSW QPS Bottleneck Diagnosis Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 精确定位 RS HNSW QPS 在 concurrency=2 时从 349 骤降到 64 的根因——是 Milvus 线程池大小限制、锁序列化、还是 CGO 调度问题。

**Architecture:** 两路并行调查：(1) 在 FFI 层加 `[HNSW_TIMING]` 诊断，测量 burst 内 peak 并发数 + burst 间 gap，在 concurrency=1 和 concurrency=2 下各跑一次对比；(2) 读 Milvus C++ 源码找调用 `HnswRustNode::Search()` 的上层代码，确认是否有线程池上限或锁。两路结果汇总后明确根因位置。

**Tech Stack:** Rust atomics (`std::sync::atomic`), `std::time::SystemTime`, hannsdb-x86 VectorDBBench, Milvus C++ 源码（hannsdb-x86）

**Background（必读）：**
- 已知数据点：
  - RS p50 搜索延迟 = 2.77ms（98% 在 L0 BFS）
  - concurrency=1: QPS=349 ≈ 1/2.87ms → 意味着 32 个 segment 全并行（若串行则 QPS≈11）
  - concurrency=2: QPS=64，[HNSW_WINDOW] 测到 peak=8, duration=27.1ms
  - 计算：31.25ms per pair - 27.1ms active = 4.15ms inter-burst gap
  - rayon 内部并行化已验证无效（QPS 349→169，reverted）
- 核心矛盾：concurrency=1 时至少 32 个 segment 并行；concurrency=2 时只有 8 个并行
- 待回答：这个 8 的上限是谁设置的？在哪里？

**Deployment:**
- 本地: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- 远端源码: `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs`
- Build cache: `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs-target`
- Milvus src: `hannsdb-x86:/data/work/milvus-rs-integ/milvus-src`
- VectorDBBench: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
- Milvus log: `/data/work/milvus-rs-integ/milvus-src/standalone.log`

---

## File Structure

- Modify: `src/ffi.rs` — 添加 `[HNSW_TIMING]` 诊断（Task 1）
- 远端只读调查，不改代码（Task 2）

---

## Task 1：在 ffi.rs 添加 [HNSW_TIMING] burst 诊断

**目的：** 精确测量每个「搜索 burst」的 peak 并发数和 burst 间 gap（死区时长），在 concurrency=1 和 concurrency=2 下各提供 20 个数据点。

**关键指标：**
- `peak`: burst 内最高同时在飞的 `knowhere_search_with_bitset` 调用数
- `burst_ms`: burst 持续时长（从第一个调用进入到最后一个调用退出）
- `gap_ms`: 上一个 burst 结束到本 burst 开始的时间（死区）

**Files:**
- Modify: `src/ffi.rs`

- [ ] **Step 1: 在 ffi.rs 文件顶部添加 import 和全局 statics**

找到 `src/ffi.rs` 的第一个 `use` 行，在文件顶부（`#![allow...]` 之后，第一个 `use` 之前）添加：

```rust
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering as FfiOrdering};

// [HNSW_TIMING] burst 并发诊断
static TIMING_IN_FLIGHT: AtomicI32 = AtomicI32::new(0);
static TIMING_PEAK: AtomicI32 = AtomicI32::new(0);
static TIMING_BURST_START_NS: AtomicU64 = AtomicU64::new(0);
static TIMING_LAST_BURST_END_NS: AtomicU64 = AtomicU64::new(0);
static TIMING_BURST_N: AtomicU64 = AtomicU64::new(0);
```

注意：如果文件中已有 `AtomicI32` 或 `AtomicU64` 的 import，只补充缺少的类型；不要重复 import。

- [ ] **Step 2: 在 knowhere_search_with_bitset 的 HNSW 分支插入诊断**

找到 `knowhere_search_with_bitset` 函数（约第 2572 行）中的 HNSW 分支：

```rust
        } else if let Some(ref idx) = index.hnsw {
            match idx.search_with_bitset_ref(query_slice, &req, &bitset_ref) {
```

替换为（在搜索调用前后插入 burst 追踪）：

```rust
        } else if let Some(ref idx) = index.hnsw {
            // [HNSW_TIMING] burst entry
            let prev_if = TIMING_IN_FLIGHT.fetch_add(1, FfiOrdering::Relaxed);
            let cur_if = prev_if + 1;
            let entry_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            if prev_if == 0 {
                TIMING_BURST_START_NS.store(entry_ns, FfiOrdering::Relaxed);
            }
            // CAS-update peak
            let mut peak = TIMING_PEAK.load(FfiOrdering::Relaxed);
            loop {
                if cur_if <= peak { break; }
                match TIMING_PEAK.compare_exchange_weak(
                    peak, cur_if, FfiOrdering::Relaxed, FfiOrdering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => peak = p,
                }
            }

            let hnsw_result = idx.search_with_bitset_ref(query_slice, &req, &bitset_ref);

            // [HNSW_TIMING] burst exit
            let remaining = TIMING_IN_FLIGHT.fetch_sub(1, FfiOrdering::Relaxed) - 1;
            if remaining == 0 {
                let end_ns = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                let burst_start = TIMING_BURST_START_NS.load(FfiOrdering::Relaxed);
                let burst_ms = end_ns.saturating_sub(burst_start) as f64 / 1_000_000.0;
                let last_end = TIMING_LAST_BURST_END_NS.swap(end_ns, FfiOrdering::Relaxed);
                let gap_ms = if last_end == 0 || burst_start < last_end {
                    0.0_f64
                } else {
                    (burst_start - last_end) as f64 / 1_000_000.0
                };
                let peak_val = TIMING_PEAK.swap(0, FfiOrdering::Relaxed);
                let n = TIMING_BURST_N.fetch_add(1, FfiOrdering::Relaxed) + 1;
                if n <= 30 || n % 200 == 0 {
                    eprintln!(
                        "[HNSW_TIMING] burst={} peak={} burst_ms={:.1} gap_ms={:.1}",
                        n, peak_val, burst_ms, gap_ms
                    );
                }
            }

            match hnsw_result {
```

注意：原来的 `match idx.search_with_bitset_ref(...)` 拆成两步。其余 match 分支（Ok/Err）代码不变，照搬。

- [ ] **Step 3: 编译验证**

```bash
cd /Users/ryan/.openclaw/workspace-builder/knowhere-rs
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

Expected: `Build OK`

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "diag(ffi): add [HNSW_TIMING] burst peak+gap diagnostic"
```

---

## Task 2：Codex 读 Milvus C++ 源码，定位搜索调度链

> **注意**: 由 Codex 在 hannsdb-x86 上执行（只读，不改代码）。与 Task 3 并行进行。

**目的：** 找到调用 `HnswRustNode::Search()` 的上层 C++ 代码，确认是否有线程池上限（如 `search_pool`、`ThreadPool`、`SemAcquire`）或排他锁。

**Files:** 只读，不改代码

- [ ] **Step 1: 找到 HnswRustNode 的 Search 实现**

```bash
ssh ryan@189.1.218.159 "cat /data/work/milvus-rs-integ/milvus-src/internal/core/thirdparty/knowhere-rs-shim/src/hnsw_rust_node.cpp"
```

重点看：
- `Search()` 函数体：是否有线程池调用？是否持锁？
- 是否调用了 `GetGlobalSearchThreadPool()` 或类似的 pool？

- [ ] **Step 2: 找调用 HnswRustNode::Search 的上层：SegmentSealed**

```bash
ssh ryan@189.1.218.159 "grep -rn 'IndexNode\|HnswRustNode\|Search.*bitset\|search_pool\|ConcurrentOp\|search_thread' \
  /data/work/milvus-rs-integ/milvus-src/internal/core/src/segcore/ \
  --include='*.cpp' --include='*.h' -l 2>/dev/null | head -20"
```

然后读最相关的文件（通常是 `SegmentSealedImpl.cpp`）：

```bash
ssh ryan@189.1.218.159 "grep -n 'Search\|thread_pool\|search_pool\|concurrent\|mutex\|lock' \
  /data/work/milvus-rs-integ/milvus-src/internal/core/src/segcore/SegmentSealedImpl.cpp \
  2>/dev/null | head -60"
```

- [ ] **Step 3: 查 knowhere 全局搜索线程池大小配置**

```bash
ssh ryan@189.1.218.159 "grep -rn 'KnowhereInitSearchThreadPool\|SetSearchThreadPoolSize\|search_thread\|global.*pool\|thread.*pool.*search' \
  /data/work/milvus-rs-integ/milvus-src/internal/core/src/ \
  --include='*.cpp' --include='*.h' 2>/dev/null | head -30"
```

重点找：实际生产路径下 pool size 是多少？是写死的还是从配置读取的？

- [ ] **Step 4: 找 Go 层 CGO 并发限制**

```bash
ssh ryan@189.1.218.159 "grep -rn 'cgo_pool\|CGOPool\|cgoPoolSize\|concurrentSearch\|SearchConcurrent\|semaphore\|Semaphore' \
  /data/work/milvus-rs-integ/milvus-src/internal/ \
  --include='*.go' 2>/dev/null | head -30"
```

- [ ] **Step 5: 写入调查结论**

把以下信息写入 `/tmp/codex_status.txt`：

```
DONE:
HnswRustNode::Search 有无锁/pool：<是/否，描述>
上层 SegmentSealedImpl 有无搜索线程池：<是/否，pool size=?>
KnowhereInitSearchThreadPool 实际值：<N>
Go CGO pool 限制：<有/无，N=?>
最可能的瓶颈位置：<描述>
```

---

## Task 3：Codex 部署 + 在 concurrency=1 和 concurrency=2 下各跑一次，提取 HNSW_TIMING 数据

> **注意**: 由 Codex 通过 tmux 在 hannsdb-x86 上执行。等 Task 1 commit 完成后发送。

**Files:** 无代码改动

- [ ] **Step 1: rsync + 远端编译**

```bash
rsync -av /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/ \
    ryan@189.1.218.159:/data/work/milvus-rs-integ/knowhere-rs/src/

ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ/knowhere-rs && \
    CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
    ~/.cargo/bin/cargo build --release 2>&1 | tail -3"
```

Expected: `Finished release [optimized]`

- [ ] **Step 2: 重启 Milvus**

```bash
ssh ryan@189.1.218.159 "pkill -f 'milvus run standalone' || true; sleep 3"
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ/milvus-src && \
    nohup ./bin/milvus run standalone > standalone.log 2>&1 &"
sleep 15
ssh ryan@189.1.218.159 "tail -3 /data/work/milvus-rs-integ/milvus-src/standalone.log"
```

- [ ] **Step 3: 运行 VectorDBBench（默认配置，即 concurrency=1）**

```bash
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ && \
    python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py \
    > /tmp/timing_c1_out.log 2>&1; echo EXIT_CODE=$?"
```

等待完成（约 20-30 分钟）。

- [ ] **Step 4: 提取 concurrency=1 的 HNSW_TIMING 数据**

```bash
ssh ryan@189.1.218.159 "grep '\[HNSW_TIMING\]' \
    /data/work/milvus-rs-integ/milvus-src/standalone.log 2>/dev/null | head -40"
```

如为空，尝试：
```bash
ssh ryan@189.1.218.159 "grep '\[HNSW_TIMING\]' /tmp/timing_c1_out.log | head -40"
```

记录 concurrency=1 的 peak 分布和 gap_ms 分布。

- [ ] **Step 5: 修改 VectorDBBench 脚本支持 concurrency=2，运行**

先检查脚本是否有 concurrency 参数：
```bash
ssh ryan@189.1.218.159 "head -80 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py | grep -i concurr"
```

如果有参数则用参数；如果是硬编码，复制脚本并改：
```bash
ssh ryan@189.1.218.159 "cp /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py \
    /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs_c2.py && \
    sed -i 's/concurrency.*=.*[0-9]\+/concurrency=2/g' \
    /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs_c2.py"
```

清空旧日志，重启 Milvus，运行 concurrency=2：
```bash
ssh ryan@189.1.218.159 "pkill -f 'milvus run standalone' || true; sleep 3"
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ/milvus-src && \
    nohup ./bin/milvus run standalone > standalone_c2.log 2>&1 &"
sleep 15
ssh ryan@189.1.218.159 "cd /data/work/milvus-rs-integ && \
    python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs_c2.py \
    > /tmp/timing_c2_out.log 2>&1; echo EXIT_CODE=$?"
```

- [ ] **Step 6: 提取 concurrency=2 的 HNSW_TIMING 数据**

```bash
ssh ryan@189.1.218.159 "grep '\[HNSW_TIMING\]' \
    /data/work/milvus-rs-integ/milvus-src/standalone_c2.log 2>/dev/null | head -40"
```

- [ ] **Step 7: 对比分析并写入结论**

对比两组数据，填写以下分析：

| 指标 | concurrency=1 | concurrency=2 | 含义 |
|------|--------------|--------------|------|
| peak (典型值) | ? | 8 | peak=32 @ c=1 → 线程池限制在 c=2 时收缩 |
| burst_ms (典型) | ? | ~27ms | burst_ms × 8/peak ≈ 2.77ms 则符合预期 |
| gap_ms (典型) | ? | ~4ms | gap 大 → 结果处理是瓶颈；gap 小 → 线程是主要瓶颈 |

分析框架：
- **若 c=1 时 peak ≈ 32, gap_ms ≈ 0**: 说明 c=1 时 Milvus 给了 32 线程，c=2 时收缩到 8。根因：某个 Milvus 配置限制了 RS 搜索总线程 = 8，在 c=1 时 1 个 query 能独占，c=2 时 2 个 query 分。
- **若 c=1 时 peak ≈ 8, burst_ms ≈ 11ms**: 说明 peak 一直是 8，c=1 时 QPS=349 的原因不是全并行，而是另有原因（如延迟更短）。
- **若 c=1 时 peak ≈ 1**: 完全序列化，c=1 QPS 靠 fast pipeline；c=2 竞争导致崩溃。

把结论写入 `/tmp/codex_status.txt`：
```
DONE: c1_peak=<N> c1_burst_ms=<X> c1_gap_ms=<Y> | c2_peak=8 c2_burst_ms=27.1 c2_gap_ms=4.1 | 结论=<一句话>
```

---

## Task 4（可选）：移除诊断代码

诊断完成后恢复干净代码。

**Files:**
- Modify: `src/ffi.rs`

- [ ] **Step 1: 移除 5 个全局 statics（TIMING_IN_FLIGHT/PEAK/BURST_START_NS/LAST_BURST_END_NS/BURST_N）**

- [ ] **Step 2: 恢复 HNSW 分支为原始形式**

```rust
        } else if let Some(ref idx) = index.hnsw {
            match idx.search_with_bitset_ref(query_slice, &req, &bitset_ref) {
```

- [ ] **Step 3: 移除不再使用的 import（AtomicI32/AtomicU64 如无其他使用）**

- [ ] **Step 4: 编译验证**

```bash
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs
git commit -m "cleanup(ffi): remove [HNSW_TIMING] diagnostic"
```
