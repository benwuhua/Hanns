# Milvus HNSW 延迟和并发瓶颈修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 RS HNSW 在 hannsdb-x86 上的两个根本问题：单次查询延迟 6.7ms vs native 3ms（2.2× 差距），以及并发 QPS 在 concurrency=10 就饱和（native 一直扩展到 concurrency=80 达 848 QPS）。

**Architecture:** 分两阶段：Phase 1 先做精确诊断（并发诊断 + 延迟分解），Phase 2 基于诊断结果打最高效益的补丁。已知高可信候选方案：Layer0Slab hugepage 分配（消除 TLB miss）、并发路径序列化点定位。

**Tech Stack:** Rust, Linux mmap/madvise, AVX512, hannsdb-x86 VectorDBBench, Milvus FFI

**Deployment Context:**
- 机器：hannsdb-x86 (189.1.218.159)，SSH alias `hannsdb-x86`
- Milvus bin: `/data/work/milvus-rs-integ/milvus-src/bin/milvus`
- RS source: `/data/work/milvus-rs-integ/knowhere-rs`
- Build cache: `/data/work/milvus-rs-integ/knowhere-rs-target`
- VectorDBBench script: `/data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py`
- Build: `CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target ~/.cargo/bin/cargo build --release`

---

## 背景分析

### 已知数据

| 指标 | RS (R4) | Native | 差距 |
|------|---------|--------|------|
| serial_latency_p99 | 6.7ms | 3ms | 2.2× slower |
| QPS @ conc=1 | 192 | 391 | 2.0× slower |
| QPS @ conc=10 | 283 | ~780+ | RS 已饱和，native 仍在增长 |
| QPS @ conc=80 | ~280 (饱和) | 848 | 3.0× gap |

RS 并发曲线：`[192, 278, 283, 283, 283, 282, 281, 280]`
Native 并发曲线：`[391, 651, 778, 794, 804, 817, 837, 848]`

### 根因假设

**延迟 (6.7ms vs 3ms)**：
- **H1 (高可信)：TLB miss 导致 Layer0Slab 读取延迟**
  - Slab 约 397MB，far beyond L3（~32MB），用标准 `Vec<u32>` 分配
  - 4KB page table → 397MB/4KB ≈ 99K 个 PTE，L1 TLB 只有 ~1500 entries → 每次 slab 随机访问高概率 TLB miss
  - 每次 HNSW 搜索访问 ~256 个节点 × 3332 bytes/节点，每个节点可能跨 page boundary
  - Linux hugepage (2MB) 修复后：397MB/2MB ≈ 199 entries → TLB 无压力
- **H2 (中可信)：Cosine 路径读取 slab 时走 slice 而非 ptr**
  - L2 快路径用 `l2_batch_4_ptrs`（raw pointer），Cosine 用 `ip_batch_4`（slice）
  - 差异本身很小，但 slice 需要 bounds check；更重要的是 Cosine 路径中 `flush remaining` 用 `self.distance()` 走非批量路径
- **H3 (低可信)：upper-layer 下降次数不同**

**并发饱和 (concurrency=10 平顶)**：
- **H4 (高可信)：某处全局序列化点**
  - RS conc=1 → conc=2 时 QPS 只从 192 → 278（1.45×，应为 2×）
  - conc=4 → conc=8 完全没提升（283 → 283）
  - 这意味着最多只有约 1.5 个有效并行 worker
  - 候选：Milvus C++ shim 侧对 RS FFI 调用加了互斥锁；或者 RS 内部某个共享写路径
- **H5 (中可信)：`borrow_mut()` 在不可重入的调用链中**
  - 如果 Milvus 同一线程内做多段搜索（nested call），TLS RefCell 会 panic 或阻塞
  - 但正常多 worker 模式下这不应该发生

---

## Phase 1：诊断

### Task 1：并发诊断仪器 — 测量 RS 侧真实并发度

**Files:**
- Modify: `src/faiss/hnsw.rs`（`search_with_bitset_ref` 函数）

这个任务在 `search_with_bitset_ref` 入口加一个原子计数器，记录同时在 RS 搜索路径中的 worker 数量，并输出统计到 stderr。运行 VectorDBBench 后查看日志即可知道 RS 侧是否真的有并发调用。

- [ ] **Step 1: 在文件顶部添加并发计数器**

在 `src/faiss/hnsw.rs` 文件顶部（现有 `thread_local!` 块附近）添加：

```rust
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};

static HNSW_CONCURRENT_SEARCHES: AtomicI32 = AtomicI32::new(0);
static HNSW_SEARCH_CALLS_TOTAL: AtomicU64 = AtomicU64::new(0);
static HNSW_PEAK_CONCURRENT: AtomicI32 = AtomicI32::new(0);
```

- [ ] **Step 2: 在 `search_with_bitset_ref` 开头插入仪器代码**

在 `pub fn search_with_bitset_ref` 函数开头（在 early return checks 之后，进入实际搜索逻辑之前）插入：

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
// Every 1000 calls, dump stats
if call_num % 1000 == 0 {
    eprintln!(
        "[HNSW_CONC] calls={} peak_concurrent={} current={}",
        call_num,
        HNSW_PEAK_CONCURRENT.load(Ordering::Relaxed),
        concurrent_now
    );
}
```

- [ ] **Step 3: 在 `search_with_bitset_ref` 函数返回前减计数**

在所有 return 语句之前，用 defer/scopeguard pattern（或在 Ok return 前）插入：

```rust
HNSW_CONCURRENT_SEARCHES.fetch_sub(1, Ordering::Relaxed);
```

由于函数有多个 return 点，最简洁的方式：创建一个 RAII guard struct：

```rust
struct ConcurrencyGuard;
impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        HNSW_CONCURRENT_SEARCHES.fetch_sub(1, Ordering::Relaxed);
    }
}
let _guard = ConcurrencyGuard;
```

将 `_guard` 放在仪器代码下面，函数结束时自动 drop。

- [ ] **Step 4: 在 hannsdb-x86 编译部署**

```bash
# 在 hannsdb-x86 执行
cd /data/work/milvus-rs-integ/knowhere-rs && git pull
CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  ~/.cargo/bin/cargo build --release 2>&1 | grep "^error" || echo "Build OK"
# 如果 Milvus 链接到 .so，需重启 Milvus
```

- [ ] **Step 5: 运行 VectorDBBench 并捕获并发日志**

```bash
# 运行 benchmark，捕获 stderr（包含并发日志）
python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py 2>&1 | tee /tmp/rs_bench_diag.log
grep "HNSW_CONC" /tmp/rs_bench_diag.log | tail -20
```

预期结果解读：
- 如果 `peak_concurrent` 始终 ≤ 2：RS 侧确实只有 ≤2 个并发调用 → **Milvus shim 侧有序列化锁**
- 如果 `peak_concurrent` ≥ 8：RS 侧确实并发，但 QPS 不扩展 → **RS 内部有锁竞争**

- [ ] **Step 6: 将诊断结果写入文件**

```bash
grep "HNSW_CONC" /tmp/rs_bench_diag.log > /tmp/codex_status.txt
# 在最后加 DONE:
echo "DONE: peak_concurrent=$(grep 'peak_concurrent' /tmp/rs_bench_diag.log | tail -1 | grep -o 'peak_concurrent=[0-9]*')" >> /tmp/codex_status.txt
```

---

### Task 2：延迟分解诊断 — 测量 search 各阶段耗时

**Files:**
- Modify: `src/faiss/hnsw.rs`（`search_single_with_bitset_ref` 函数）

在 `search_single_with_bitset_ref` 中对各阶段计时，每 500 次查询输出一次 p50/p99。

- [ ] **Step 1: 在文件顶部添加延迟统计 TLS**

```rust
thread_local! {
    static HNSW_LATENCY_SAMPLES: RefCell<Vec<[u64; 4]>> = RefCell::new(Vec::new());
    // [upper_descent_ns, l0_search_ns, finalize_ns, total_ns]
}
```

- [ ] **Step 2: 在 `search_single_with_bitset_ref` 中插入计时**

```rust
fn search_single_with_bitset_ref<'a>(...) -> Vec<(i64, f32)> {
    let t0 = std::time::Instant::now();

    // ... existing early returns ...

    self.prepare_search_query_context(query);
    let mut curr_ep_idx = ...;

    let t1 = std::time::Instant::now();
    // upper layer descent
    let mut curr_ep_dist = self.distance(query, curr_ep_idx);
    for level in ... {
        ...
    }
    let t2 = std::time::Instant::now();

    let mut result = HNSW_SEARCH_SCRATCH_TLS.with(|scratch_cell| {
        let mut scratch = scratch_cell.borrow_mut();
        self.search_single_with_bitset_ref_scratch(...)
    });
    let t3 = std::time::Instant::now();

    self.clear_search_query_context();
    self.rerank_sq_results(query, &mut result);
    let t4 = std::time::Instant::now();

    // Record timing sample
    let upper_ns = t2.duration_since(t1).as_nanos() as u64;
    let l0_ns = t3.duration_since(t2).as_nanos() as u64;
    let final_ns = t4.duration_since(t3).as_nanos() as u64;
    let total_ns = t4.duration_since(t0).as_nanos() as u64;

    HNSW_LATENCY_SAMPLES.with(|cell| {
        let mut v = cell.borrow_mut();
        v.push([upper_ns, l0_ns, final_ns, total_ns]);
        if v.len() >= 500 {
            let mut samples = v.clone();
            v.clear();
            drop(v);
            // Compute and print percentiles
            samples.sort_by_key(|x| x[3]);
            let n = samples.len();
            let p50 = &samples[n / 2];
            let p99 = &samples[n * 99 / 100];
            eprintln!(
                "[HNSW_LAT] p50: upper={:.0}us l0={:.0}us fin={:.0}us total={:.0}us | p99: upper={:.0}us l0={:.0}us fin={:.0}us total={:.0}us",
                p50[0] as f64 / 1000.0, p50[1] as f64 / 1000.0, p50[2] as f64 / 1000.0, p50[3] as f64 / 1000.0,
                p99[0] as f64 / 1000.0, p99[1] as f64 / 1000.0, p99[2] as f64 / 1000.0, p99[3] as f64 / 1000.0,
            );
        }
    });

    result
}
```

- [ ] **Step 3: 编译部署并运行**

与 Task 1 相同的部署步骤。运行后查看：

```bash
grep "HNSW_LAT" /tmp/rs_bench_diag.log | tail -5
```

预期结果解读：
- 如果 l0_ns 占比 >90%：主要时间在 L0 搜索（内积计算 + 内存访问）
- 如果 upper_ns >500us：上层下降异常慢
- 如果 total_ns >> upper+l0+fin：有其他开销（FFI overhead、borrow_mut、etc.）

- [ ] **Step 4: 将诊断结果写入文件，供 Claude review**

```bash
echo "DONE: $(grep 'HNSW_LAT' /tmp/rs_bench_diag.log | tail -3)" > /tmp/codex_status.txt
```

---

## Phase 2：针对性修复

> **注意**：Phase 2 的 Task 3/4/5 应在 Phase 1 诊断结果分析后决定执行顺序。以下给出最高可信度的修复先行。

### Task 3：Layer0Slab Hugepage 分配（最高优先级修复）

**依据**：H1 假设 — TLB miss 是 397MB Slab 延迟的主因。Linux `madvise(MADV_HUGEPAGE)` 可请求内核将 slab 映射到 2MB 大页，TLB 压力从 99K entries → 199 entries。

**预期效果**：基于 TLB miss 理论分析，如果 slab 访问确实是延迟主因，hugepage 可将 L0 搜索延迟降低 30-60%，可能将 6.7ms → 3-4ms。

**Files:**
- Modify: `src/faiss/hnsw.rs`（`Layer0Slab` 的分配函数 + `rebuild_layer0_slab`）

- [ ] **Step 1: 定位 Layer0Slab 分配位置**

在 `hnsw.rs` 中找到 `Layer0Slab` struct 和 `rebuild_layer0_slab`（或相应的 build 函数），找到分配 `words: Vec<u32>` 的地方。用 `grep -n "Layer0Slab\|rebuild_layer0_slab\|slab.*Vec\|Vec.*u32.*slab" src/faiss/hnsw.rs` 定位。

- [ ] **Step 2: 添加 madvise hugepage 辅助函数**

在文件顶部添加：

```rust
/// 对 Vec<u32> 的底层内存调用 madvise(MADV_HUGEPAGE)。
/// 这是一个 hint，内核可能忽略。失败时不影响正确性。
#[cfg(target_os = "linux")]
fn madvise_hugepage<T>(data: &[T]) {
    use std::os::raw::{c_int, c_void};
    extern "C" {
        fn madvise(addr: *mut c_void, length: usize, advice: c_int) -> c_int;
    }
    const MADV_HUGEPAGE: c_int = 14;
    if data.is_empty() {
        return;
    }
    let ptr = data.as_ptr() as *mut c_void;
    let len = data.len() * std::mem::size_of::<T>();
    // madvise 要求页对齐；Vec 的内存由 allocator 保证对齐
    unsafe {
        madvise(ptr, len, MADV_HUGEPAGE);
    }
}

#[cfg(not(target_os = "linux"))]
fn madvise_hugepage<T>(_data: &[T]) {}
```

- [ ] **Step 3: 在 Slab 分配完成后调用 madvise**

在 `Layer0Slab` 的构建函数中，`words` Vec 分配并填充完毕后，添加：

```rust
// 请求内核将 slab 映射到 2MB 大页，消除 TLB miss
// 约 397MB @ 119K nodes，标准 4KB page 需 ~99K TLB entries，2MB 大页只需 ~199 entries
madvise_hugepage(&self.words);
```

具体位置：在 `self.words = vec![0u32; total_words];` 填充完 neighbor 数据后（`rebuild_layer0_slab` 末尾）。

- [ ] **Step 4: 也对 `flat_graph` 数据调用 madvise（如果存在）**

同样在 `layer0_flat_graph` 的底层 Vec 数据上调用 `madvise_hugepage`。

- [ ] **Step 5: 在 hannsdb-x86 编译验证**

```bash
cd /data/work/milvus-rs-integ/knowhere-rs
git pull  # 或手动 rsync 修改
CARGO_TARGET_DIR=/data/work/milvus-rs-integ/knowhere-rs-target \
  ~/.cargo/bin/cargo build --release 2>&1 | grep "^error" || echo "Build OK"
```

- [ ] **Step 6: 确认 hugepage 是否生效（Linux 系统检查）**

```bash
# 确认 /sys/kernel/mm/transparent_hugepage/enabled 是 [always] 或 [madvise]
cat /sys/kernel/mm/transparent_hugepage/enabled
# 如果是 [never]，需要 sudo echo madvise > /sys/kernel/mm/transparent_hugepage/enabled
# 运行 Milvus 后检查 huge pages 使用量
grep -i "hugepage\|huge" /proc/meminfo | head -10
```

- [ ] **Step 7: 运行 VectorDBBench 并对比延迟数字**

运行与 R4 相同的 benchmark 脚本，记录新的 QPS、延迟数据。

```bash
python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py 2>&1 | tee /tmp/rs_hugepage_bench.log
grep -E "QPS|latency|recall" /tmp/rs_hugepage_bench.log | tail -20
echo "DONE: <填入新 QPS 数字>" > /tmp/codex_status.txt
```

---

### Task 4：修复 Milvus shim 侧并发序列化（依赖 Task 1 诊断结果）

**仅当 Task 1 诊断显示 `peak_concurrent ≤ 2` 时执行此任务。**

如果 RS 侧只有 ≤2 个并发调用，说明 Milvus C++ shim 在调用 RS FFI 时有互斥锁。需要定位并修复 shim 代码。

**Files:**
- Read: Milvus knowhere shim（`/data/work/milvus-rs-integ/knowhere/src/index/` 中 HnswRs 相关文件）
- Modify: shim 中对应的搜索函数

- [ ] **Step 1: 定位 Milvus C++ shim 搜索路径**

```bash
ssh hannsdb-x86 "find /data/work/milvus-rs-integ/knowhere -name '*.cc' -o -name '*.cpp' | xargs grep -l 'knowhere_search_with_bitset\|HnswRs\|rs_hnsw' 2>/dev/null"
```

- [ ] **Step 2: 检查搜索路径中是否有互斥锁**

```bash
ssh hannsdb-x86 "grep -n 'lock\|mutex\|shared_mutex\|unique_lock\|lock_guard' \
  <找到的文件路径> | head -30"
```

- [ ] **Step 3: 如果发现写锁（`unique_lock` 或 `lock_guard`）用于搜索**

将写锁改为读锁（`shared_lock`）：

```cpp
// 修改前（序列化）：
std::unique_lock<std::shared_mutex> lock(mutex_);
auto result = knowhere_search_with_bitset(index_, ...);

// 修改后（允许并发读）：
std::shared_lock<std::shared_mutex> lock(mutex_);
auto result = knowhere_search_with_bitset(index_, ...);
```

- [ ] **Step 4: 重新编译 Milvus 并运行 benchmark 验证**

```bash
# 在 hannsdb-x86 重新编译 knowhere 库并重建 Milvus
cd /data/work/milvus-rs-integ/milvus-src && make build_knowhere_rs 2>&1 | tail -5
python3 /data/work/VectorDBBench/run_milvus_hnsw_1m_cohere_rs.py 2>&1 | tee /tmp/rs_shim_fix_bench.log
echo "DONE: <QPS 数字>" > /tmp/codex_status.txt
```

---

### Task 5：RS 内部并发检查（依赖 Task 1 诊断结果）

**仅当 Task 1 诊断显示 `peak_concurrent ≥ 8` 但 QPS 仍不扩展时执行。**

说明 RS 侧确实有并发调用，但内部有某个共享写状态。

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: 审查 search 路径上所有共享写状态**

```bash
grep -n "static.*AtomicU64\|static.*AtomicI32\|lazy_static\|OnceLock" src/faiss/hnsw.rs
```

目前已知的全局写：
- `HNSW_PEAK_CONCURRENT`（Task 1 新增，仅统计用，无争用问题）
- 无其他已知全局写状态

- [ ] **Step 2: 检查 `node_info` 等字段是否在 search 期间有隐式写**

在 `search_single_with_bitset_ref` 调用链中，所有访问的 `self.xxx` 字段应为只读（`&self` receiver）。验证是否有 `Cell<>` 或 `RefCell<>` 在 struct 字段上被意外写入：

```bash
grep -n "Cell\|RefCell" src/faiss/hnsw.rs | grep "struct\|pub\|    [a-z]" | head -20
```

- [ ] **Step 3: 如果发现争用点，添加细粒度锁或改为无锁设计**

具体修复取决于发现内容。

---

## 执行顺序和决策点

```
Task 1 (并发诊断) → 分析 peak_concurrent
  ├─ peak ≤ 2 → Task 4 (修复 shim 锁) + Task 3 (hugepage)
  └─ peak ≥ 8 → Task 3 (hugepage) + Task 5 (内部锁)

Task 2 (延迟分解) → 分析各阶段耗时
  ├─ l0_ns 占比 >90% → Task 3 (hugepage) 是正确方向
  └─ upper_ns 或 finalize_ns 异常 → 针对性修复
```

Tasks 1 和 2 可以同时执行（同一次 VectorDBBench 运行即可收集两者数据）。
Task 3 独立于诊断，可与 Task 1/2 并行实施（需要分别在两次运行中验证）。

---

## 预期目标

| 修复 | 预期改善 |
|------|---------|
| Hugepage (Task 3) | 延迟 6.7ms → ~3-4ms（如 TLB 是主因）|
| Shim 锁修复 (Task 4) | QPS 并发扩展 10+ → 80，达到 native 水平 |
| 两者结合 | QPS @ conc=80 接近或超过 native 848 |

**最终目标**：在 hannsdb-x86 Cohere-1M 上，RS QPS 超过 native 848，完成 HNSW 领先地位的最终验证。
