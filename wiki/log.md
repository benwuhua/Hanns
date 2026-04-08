# Log — knowhere-rs

append-only 时间线。新条目加在顶部。

---

## 2026-04-08 — IVF-SQ8 Milvus 集成完成

**类型**：feat
**Commits**：`17d8b90` (Rust FFI) + shim commits `58bda35`, `3d7e6ce`, `0c4b6f5`
**结果**：
- `knowhere_set_nprobe` FFI 新增，IVF 搜索路径 nprobe 动态化
- C++ shim `ivf_sq8_rust_node.cpp` 完整实现（Train/Add/Search/Serialize/Deserialize）
- Milvus `INDEX_FAISS_IVFSQ8` float32 路径拦截到 RS 实现
- Smoke test ✅，serial QPS=39（100K×768D，H 开销主导）
- 等待 Cohere-1M 权威对比（数据未缓存）

---

## 2026-04-08 — DiskANN PQ 参数通道端到端验证

**类型**：验证
**结果**：NoPQ + PQ(`pq_budget=0.002`) 两模式 Milvus 全链路 OK，top-distance 完全一致 ✅
→ `pq_code_budget_gb` / `beamwidth` 参数从 pymilvus → C++ shim → Rust FFI 全路径贯通

---

## 2026-04-08 — DiskANN 参数对齐修复

**类型**：feat + fix
**Commits**：`a1e49ba` (Rust) + `e231450` + `d6eb0f2` (C++ shim)

DiskANN 参数通道补全：`pq_code_budget_gb` / `build_dram_budget_gb` / `disk_pq_dims` / `beamwidth` 从 Milvus 侧打通到 RS `AisaqConfig`。Shim fallback 值修正（max_degree 56→48，search_list_size 100→128，beamwidth 改为 `value_or(8)`）。

→ 详见 [[decisions/materialize-storage]] §参数对齐

---

## 2026-04-08 — DiskANN Milvus R2：materialize_storage fix

**类型**：perf
**Commits**：`72e1cd8` (fix) + `7f3662b` (bench)

**根因**：`PQFlashIndex::load()` 始终创建 `storage: Some(DiskStorage)`，即使 NoPQ in-memory 模式，导致 search 走 PageCache 路径（Arc alloc + memcpy per node）。

**修复**：`load()` 末尾加 `if pq_code_size == 0 { materialize_storage()?; }`

**结果**：
- Milvus DiskANN serial QPS：2.4 → **11.2**（+4.7×）
- 对比 native：11.2 vs 11.4（parity，误差范围内）
- 100K per-segment：RS 41.3 vs native 36.6（RS **+13%**）

→ 详见 [[decisions/materialize-storage]] | [[benchmarks/diskann-milvus-rounds]]

---

## 2026-04-07 — DiskANN Milvus R1：FFI 全部接通

**类型**：feat
**Commits**：`52bdf81`…`bc5ac75` (FFI wiring series)

`CIndexType::DiskAnn` 在 `src/ffi.rs` 完整接通：new/train/add/search/search_with_bitset/save/load。C++ shim 新增 `DiskAnnRustNode`（`diskann_rust_node.cpp`），`index_factory.h` 路由 `INDEX_DISKANN → MakeDiskAnnRustNode()`。

**诊断验证**：100K×768D，DiskANN 36.6 QPS vs FLAT 11.2 QPS = **3.27×**，确认 Vamana graph search 生效（非暴力搜索）。

**R1 问题**：1M serial QPS = 2.4（与 FLAT 相近）→ 根因在 R2 修复。

→ 详见 [[benchmarks/diskann-milvus-rounds]]

---

## 2026-04-07 — HNSW Milvus R8：alloc-reduction，超越 native

**类型**：perf
**Commits**：`cd5625f` + `0a6d699`

两处关键优化：
1. `to_sorted_pairs()` clone 消除：`clone().into_sorted_vec()` → `mem::take().into_sorted_vec()`
2. 预分配 flat 输出 buffer：`Vec<Mutex<Vec>>` + 后处理 copy → `SendPtr` raw pointer 写入不相交切片

**结果**：c=20: 548→**1051** QPS (+92%)；c=80: 540→**1042** QPS (+93%)。RS 超越 native ~800 QPS。

→ 详见 [[benchmarks/hnsw-milvus-rounds]] | [[decisions/optimization-log]]

---

## 2026-04-07 — HNSW Milvus R7：私有 rayon pool

**类型**：perf
**Commit**：`06d3ec2`

`HNSW_NQ_POOL`：`once_cell::Lazy<rayon::ThreadPool>`，大小 `(hw - 32).max(4)`。CGO executor 线程调用 `install()` 加入私有池，`scope_fifo()` 保证 lifetime 安全。

**结果**：c=20: 91→548 QPS (+6×)；c=80: 349→540 QPS (+55%)

→ 详见 [[decisions/rayon-private-pool]]

---

## 2026-04-06 — HNSW R6 rayon revert（失败案例）

**类型**：revert
**Commit**：`89e99f3`（已 revert）

尝试在 nq 循环用 `into_par_iter()` 并行：QPS 349→170 (−51%)，Optimize 336.9s→4464s (13×)。根因：CGO executor 线程 JOIN rayon 全局池 → 过度订阅。

→ 详见 [[decisions/rayon-private-pool]]

---

## 2026-04-05 — HNSW Milvus hugepage (R5)

**类型**：perf
**Commit**：`c5db83f`

`madvise(MADV_HUGEPAGE)` 对 Layer0Slab（397MB）：4KB→2MB 页，TLB entries 99K→199。延迟 6.7ms→2.8ms（2.4×），QPS 不变（瓶颈在 Milvus query serialization，非搜索速度）。

---

## 2026-04-04 — HNSW Milvus R4：FFI lazy bitset（parity 达成）

**类型**：perf
**Commit**：`106dee7`

FFI 层 bitset 分配从无条件改为按需：`bitset_view`（~15KB）+`sparse_bitset`（~120KB）移入实际需要的分支。HNSW 走 zero-copy `bitset_ref`，节省 107MB/s 无用堆分配。

**结果**：285→349 QPS (+22%)，达到 native parity。

---

## 2026-03-26 — AISAQ SIFT-1M 权威数字确立

**类型**：bench
**Commit**：`abf7aea`（recall+build 双修）

NoPQ 内存模式：recall 0.9941，QPS 6,365，build 244s（native 1595s，**6.5× faster build**）。
disk/mmap 冷启动：401 QPS。PQ32 disk V3 group=8：1,063 QPS，recall 0.9114。

→ 详见 [[benchmarks/authority-numbers]]
