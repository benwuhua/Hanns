# Milvus HNSW Insert 优化计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 RS HNSW Insert 时间从 336.8s 降到低于 native 304.6s，消除 +10.6% 差距。

**Architecture:** 先通过诊断仪器确认 Insert 阶段是否调用 `knowhere_add_index` 及 batch 规模；根据结果应用最高效的修复（并行 add 或 FFI-layer batch 累积器）。

**Tech Stack:** Rust, src/ffi.rs, src/faiss/hnsw.rs, hannsdb-x86 VectorDBBench

**Background:**
- 当前状态：RS Insert=336.8s，native Insert=304.6s（+10.6% 慢）
- 并行 HNSW add 存在但需要 `KNOWHERE_RS_FFI_ENABLE_PARALLEL_HNSW_ADD=1` env var + batch ≥1000 + num_threads>1
- 不清楚 Milvus Insert 阶段是否实际调用 knowhere，以及 batch size 是多少
- Optimize 时间 RS 已经 2.53× 快于 native（363s vs 854s），说明 HNSW 批量构建本身是快的

**Deployment:**
- 本地代码: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs`
- 远端: `hannsdb-x86:/data/work/milvus-rs-integ/knowhere-rs`
- Build: `CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target ~/.cargo/bin/cargo build --release`
- Benchmark: `python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py 2>/tmp/rs_insert_diag.log`

---

## File Structure

- Modify: `src/ffi.rs` — `knowhere_add_index` 函数（诊断 + batch 累积器）
- Modify: `src/ffi.rs` — `IndexWrapper` struct（累积 buffer field，如果需要）

---

## Task 1：添加 Insert 诊断仪器到 knowhere_add_index

**Purpose:** 确认两个未知量：
1. Milvus Insert 阶段是否调用 `knowhere_add_index`（如果不调用，Insert 慢的原因在 Milvus 层，不在 knowhere）
2. 调用时的 batch size 分布（决定是否可用并行 add）

**Files:**
- Modify: `src/ffi.rs`

- [ ] **Step 1: 在文件顶部添加 Insert 诊断全局变量**

在 `src/ffi.rs` 文件中已有 `use std::sync::atomic` 的地方（或在顶部添加），插入：

```rust
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static HNSW_ADD_CALLS: AtomicU64 = AtomicU64::new(0);
static HNSW_ADD_VECTORS_TOTAL: AtomicU64 = AtomicU64::new(0);
static HNSW_ADD_TIME_NS: AtomicU64 = AtomicU64::new(0);
```

注意：如果 `AtomicU64` 或 `AtomicOrdering` 已经在文件中 import，不要重复 import，只添加缺少的部分。

- [ ] **Step 2: 在 `knowhere_add_index` 函数中插入诊断代码**

在 `knowhere_add_index` 函数的 `match index.add(vectors_slice, ids_slice)` 调用前后插入：

```rust
pub extern "C" fn knowhere_add_index(
    index: *mut std::ffi::c_void,
    vectors: *const f32,
    ids: *const i64,
    count: usize,
    dim: usize,
) -> i32 {
    if index.is_null() || vectors.is_null() || count == 0 || dim == 0 {
        return CError::InvalidArg as i32;
    }

    unsafe {
        let index = &mut *(index as *mut IndexWrapper);

        let vectors_slice = std::slice::from_raw_parts(vectors, count * dim);
        let ids_slice = if !ids.is_null() {
            Some(std::slice::from_raw_parts(ids, count))
        } else {
            None
        };

        // INSERT DIAGNOSTIC: count calls, batch sizes, elapsed time
        let t0 = std::time::Instant::now();
        let result = match index.add(vectors_slice, ids_slice) {
            Ok(_) => CError::Success as i32,
            Err(e) => e as i32,
        };
        let elapsed_ns = t0.elapsed().as_nanos() as u64;

        let call_n = HNSW_ADD_CALLS.fetch_add(1, AtomicOrdering::Relaxed) + 1;
        HNSW_ADD_VECTORS_TOTAL.fetch_add(count as u64, AtomicOrdering::Relaxed);
        HNSW_ADD_TIME_NS.fetch_add(elapsed_ns, AtomicOrdering::Relaxed);

        // Log every 100 calls: call_n, batch_size, cumulative_vectors, cumulative_ms
        if call_n % 100 == 0 || call_n <= 5 {
            eprintln!(
                "[HNSW_ADD] call={} batch_size={} total_vectors={} total_ms={:.0}",
                call_n,
                count,
                HNSW_ADD_VECTORS_TOTAL.load(AtomicOrdering::Relaxed),
                HNSW_ADD_TIME_NS.load(AtomicOrdering::Relaxed) as f64 / 1_000_000.0,
            );
        }

        result
    }
}
```

- [ ] **Step 3: 编译验证**

```bash
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

Expected: `Build OK`

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "diag(ffi): add [HNSW_ADD] insert timing diagnostic to knowhere_add_index"
```

---

## Task 2：部署到 hannsdb-x86 并运行诊断（Codex 负责）

> **注意**: 这个 task 由 Codex 通过 tmux 在 hannsdb-x86 上执行，不是 Agent subagent。

- [ ] rsync 本地 src/ 到 hannsdb-x86
- [ ] 远端 `cargo build --release`
- [ ] 运行 VectorDBBench，捕获 stderr
- [ ] 提取 `[HNSW_ADD]` 日志行
- [ ] 分析关键问题：
  - 是否有 `[HNSW_ADD]` 输出？（如果没有，Insert 阶段不调用 knowhere）
  - batch_size 是多少？（如果 ≥1000，可以用并行 add）
  - total_ms 是多少？（knowhere add 占 Insert 总时间多少比例）

**预期解读：**

| 情况 | 含义 | 下一步 |
|------|------|--------|
| 无 `[HNSW_ADD]` 输出 | Insert 不调用 knowhere | 查 Milvus Go 层，Insert 慢的原因在 data pipeline |
| batch_size ≥1000，total_ms 占 Insert 大部分 | 可以并行化 | Task 3A |
| batch_size <1000（小批次多次调用） | 需要 batch 累积器 | Task 3B |
| total_ms 很小（<10%的Insert时间） | knowhere 不是瓶颈 | 无法在 knowhere 侧优化 |

---

## Task 3A：启用并行 HNSW add（如果 batch_size ≥1000）

**Condition:** Task 2 诊断显示 batch_size ≥1000

**原因:** 当前 `should_use_parallel_hnsw_add_via_ffi` 需要 `FFI_ENABLE_PARALLEL_HNSW_ADD=1` env var，而 Milvus 启动时可能未设置该变量。

**Files:**
- Modify: `src/ffi.rs` — `should_use_parallel_hnsw_add_via_ffi` 函数

**方案：修改默认策略 — 当 batch 足够大时，默认启用并行 add，不需要 env var**

- [ ] **Step 1: 修改 `should_use_parallel_hnsw_add_via_ffi` 默认逻辑**

找到 `src/ffi.rs` 中的 `should_use_parallel_hnsw_add_via_ffi` 函数：

```rust
fn should_use_parallel_hnsw_add_via_ffi(idx: &HnswIndex, count: usize) -> bool {
    !Self::ffi_force_serial_hnsw_add()
        && Self::ffi_enable_parallel_hnsw_add()
        && idx.should_use_parallel_add(count)
}
```

改为：

```rust
fn should_use_parallel_hnsw_add_via_ffi(idx: &HnswIndex, count: usize) -> bool {
    // Default: auto-enable parallel for large batches unless explicitly forced serial.
    // The env var opt-in is no longer required; parallel is used whenever the batch
    // size and thread count justify it. Set KNOWHERE_RS_FFI_FORCE_SERIAL_HNSW_ADD=1
    // to disable (e.g., for single-threaded benchmarks or debugging).
    !Self::ffi_force_serial_hnsw_add() && idx.should_use_parallel_add(count)
}
```

- [ ] **Step 2: 验证测试仍通过**

```bash
cargo test --test ffi_tests 2>&1 | tail -5
# 或者如果测试文件名不同：
cargo test ffi_enable_parallel 2>&1 | tail -5
```

Expected: 相关测试通过（如果有测试验证串行 add 行为，检查是否需要更新预期）

- [ ] **Step 3: 编译**

```bash
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "perf(ffi): auto-enable parallel HNSW add for large batches without env var"
```

---

## Task 3B：FFI-layer batch 累积器（如果 batch_size <1000，多次小批次调用）

**Condition:** Task 2 诊断显示 batch_size 远小于 1000，且 total_ms 占 Insert 时间的显著比例

**原因:** 小批次（如 <200 vectors/call）无法触发并行 add。需要在 FFI 层累积 vectors，凑满阈值后一次性 add，减少每次 add 的固定开销（graph ptr 更新、entry_point 判断等）。

**Files:**
- Modify: `src/ffi.rs` — `IndexWrapper` struct 添加 accumulation buffer
- Modify: `src/ffi.rs` — `knowhere_add_index` 函数
- Modify: `src/ffi.rs` — `knowhere_build_index` 或等效的 flush 触发点

**注意:** Task 3B 的实现比 Task 3A 复杂得多，且只在诊断确认小批次场景后执行。

- [ ] **Step 1: 在 `IndexWrapper` struct 添加 accumulation buffer**

找到 `IndexWrapper` struct 定义，添加字段：

```rust
struct IndexWrapper {
    // ... existing fields ...
    hnsw: Option<HnswIndex>,
    // ... other existing fields ...

    // Accumulation buffer for batching small FFI add calls
    // Only used for HNSW; other index types add immediately.
    hnsw_add_buffer_vectors: Vec<f32>,
    hnsw_add_buffer_ids: Vec<i64>,
    hnsw_add_buffer_threshold: usize,  // flush when buffer reaches this size
}
```

在 `IndexWrapper::new` 或构造函数中初始化（threshold=4096 or configurable）：

```rust
hnsw_add_buffer_vectors: Vec::new(),
hnsw_add_buffer_ids: Vec::new(),
hnsw_add_buffer_threshold: 4096,
```

- [ ] **Step 2: 修改 `IndexWrapper::add` 中 HNSW 分支**

在 `add()` 函数的 `hnsw` 分支中，先将 vectors 推入 buffer，当 buffer ≥ threshold 时 flush：

```rust
} else if let Some(ref mut idx) = self.hnsw {
    let dim = idx.dim();

    // Accumulate vectors
    self.hnsw_add_buffer_vectors.extend_from_slice(vectors);
    if let Some(ids) = ids {
        self.hnsw_add_buffer_ids.extend_from_slice(ids);
    } else {
        // Generate sequential IDs from current index size
        let start = idx.len() + self.hnsw_add_buffer_vectors.len() / dim - vectors.len() / dim;
        let n = vectors.len() / dim;
        self.hnsw_add_buffer_ids.extend((start as i64..(start + n) as i64));
    }

    // Flush when buffer is large enough
    let buf_count = self.hnsw_add_buffer_vectors.len() / dim;
    if buf_count >= self.hnsw_add_buffer_threshold {
        let result = if Self::should_use_parallel_hnsw_add_via_ffi(idx, buf_count) {
            idx.add_parallel(&self.hnsw_add_buffer_vectors, Some(&self.hnsw_add_buffer_ids), Some(true))
                .or_else(|_| idx.add(&self.hnsw_add_buffer_vectors, Some(&self.hnsw_add_buffer_ids)))
        } else {
            idx.add(&self.hnsw_add_buffer_vectors, Some(&self.hnsw_add_buffer_ids))
        };
        self.hnsw_add_buffer_vectors.clear();
        self.hnsw_add_buffer_ids.clear();
        result.map(|_| buf_count).map_err(|_| CError::Internal)
    } else {
        Ok(vectors.len() / dim)
    }
}
```

- [ ] **Step 3: 添加 `flush_hnsw_buffer` 函数并在 build_index 调用**

```rust
fn flush_hnsw_buffer(&mut self) -> Result<usize, CError> {
    if self.hnsw_add_buffer_vectors.is_empty() {
        return Ok(0);
    }
    if let Some(ref mut idx) = self.hnsw {
        let dim = idx.dim();
        let buf_count = self.hnsw_add_buffer_vectors.len() / dim;
        let result = if Self::should_use_parallel_hnsw_add_via_ffi(idx, buf_count) {
            idx.add_parallel(&self.hnsw_add_buffer_vectors, Some(&self.hnsw_add_buffer_ids), Some(true))
                .or_else(|_| idx.add(&self.hnsw_add_buffer_vectors, Some(&self.hnsw_add_buffer_ids)))
        } else {
            idx.add(&self.hnsw_add_buffer_vectors, Some(&self.hnsw_add_buffer_ids))
        };
        self.hnsw_add_buffer_vectors.clear();
        self.hnsw_add_buffer_ids.clear();
        result.map(|_| buf_count).map_err(|_| CError::Internal)
    } else {
        Ok(0)
    }
}
```

在 `knowhere_build_index`（或 Milvus 调用的等效 finalize 函数）中，在 actual build 之前调用 `flush_hnsw_buffer()`。

也在 `knowhere_search_with_bitset` 入口添加 flush（确保搜索前所有 vectors 都已加入）：

```rust
// Flush any pending buffer before search
if let Some(_) = index.hnsw.as_ref() {
    let _ = index.flush_hnsw_buffer();
}
```

- [ ] **Step 4: 测试 buffer 不丢数据**

```bash
# 运行 HNSW 相关测试确保 add/search 正确
cargo test hnsw --lib 2>&1 | tail -10
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/ffi.rs
git commit -m "perf(ffi): batch HNSW add calls to reduce small-batch FFI overhead"
```

---

## Task 4：移除诊断代码（Task 1 添加的）

诊断目的完成后，移除 `[HNSW_ADD]` 仪器代码，恢复干净的 `knowhere_add_index` 实现。

**Files:**
- Modify: `src/ffi.rs`

- [ ] **Step 1: 移除 3 个 AtomicU64 全局变量（HNSW_ADD_CALLS/VECTORS_TOTAL/TIME_NS）**

- [ ] **Step 2: 恢复 `knowhere_add_index` 为无诊断的干净版本**

恢复为（保留 Task 3A 或 Task 3B 的修改，只移除诊断部分）：

```rust
pub extern "C" fn knowhere_add_index(
    index: *mut std::ffi::c_void,
    vectors: *const f32,
    ids: *const i64,
    count: usize,
    dim: usize,
) -> i32 {
    if index.is_null() || vectors.is_null() || count == 0 || dim == 0 {
        return CError::InvalidArg as i32;
    }

    unsafe {
        let index = &mut *(index as *mut IndexWrapper);

        let vectors_slice = std::slice::from_raw_parts(vectors, count * dim);
        let ids_slice = if !ids.is_null() {
            Some(std::slice::from_raw_parts(ids, count))
        } else {
            None
        };

        match index.add(vectors_slice, ids_slice) {
            Ok(_) => CError::Success as i32,
            Err(e) => e as i32,
        }
    }
}
```

- [ ] **Step 3: 编译**

```bash
cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

- [ ] **Step 4: Commit**

```bash
git add src/ffi.rs
git commit -m "cleanup(ffi): remove HNSW_ADD diagnostic instrumentation"
```

---

## Execution Order

```
Task 1 (诊断代码) → Codex 部署到 hannsdb-x86 → 分析 [HNSW_ADD] 输出
  ├─ 无 HNSW_ADD 输出 → Insert 不调用 knowhere，停止，记录结论
  ├─ batch_size ≥1000 → Task 3A (启用并行 add)
  └─ batch_size <1000 → Task 3B (batch 累积器)
Task 4 (清理诊断代码) — 在 Task 3A 或 3B 完成后执行
```

**目标指标:** Insert ≤ 304.6s (native baseline) on hannsdb-x86 Cohere-1M HNSW M=16
