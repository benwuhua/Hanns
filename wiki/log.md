# Log — knowhere-rs

append-only 时间线。新条目加在顶部。

---

## 2026-04-08 — KMeans 预分配优化（build time 大逆转）

**类型**：perf optimization

**根因**：`train_parallel()` 每轮迭代（默认 25 次）重新分配：
- `new_centroids: Vec<f32>[k×dim]` = 3.1MB（k=1024, dim=768）
- `counts: Vec<usize>[k]`
- 每个 centroid 单独 `updated: Vec<f32>[dim]` × 1024 次微分配

**修复**：
1. `assignments`/`new_centroids`/`counts` 移到循环外预分配，循环内 `.fill(0)` 复用
2. 消除 per-centroid `updated` 分配，in-place 计算归一化
3. 用 `collect_into_vec` 写回预分配 assignments（保持 rayon 并行）

**结果**（100K×768D Milvus 对比）：
- IVF_FLAT build: RS 5.5→**3.0s**，Native 3.0→4.5s → **RS 1.5× 更快** ✅
- IVF_SQ8 build: RS 6.0→**5.5s**，Native 3.0→7.5s → **RS 1.36× 更快** ✅
- IVF_PQ build: RS 7.0→**3.6s**，Native 3.5→3.0s → parity ✅

从 "RS 比 native 慢 2×" 翻转为 "RS 比 native 快 1.36–1.5×"。

---

## 2026-04-08 — IVF 系列 Build Time + Recall 完整对比

**类型**：bench

**方法**：每个 index drop + rebuild，RS vs Native 各建一次，测 build time / recall / QPS。

**发现**：
- Recall 完全 parity ✅
- c=1 search QPS parity ✅
- **Build time：RS ~2× 慢于 native ⚠️**（IVF_FLAT: 5.5s vs 3.0s，IVF_SQ8: 6.0s vs 3.0s，IVF_PQ: 7.0s vs 3.5s）

Build 差距来源：k-means 训练（IVF 聚类）。100K 数据绝对差距小（4s 以内），但 1M 数据可能扩大到 40s 量级。需要后续优化。

→ 详见 [[benchmarks/authority-numbers]] §IVF Build Time

---

## 2026-04-08 — IVF-PQ Milvus 集成

**类型**：集成 + bench

**改动**：
- `src/ffi.rs`：`CIndexConfig` 添加 `pq_m`/`pq_nbits` 字段；`IvfPq` 分支传入 `m`/`nbits_per_idx`
- 新建 `ivf_pq_rust_node.cpp` shim（读取 Milvus config 中 `m`/`nbits` key）
- `cabi_bridge.hpp` 同步添加 `pq_m`/`pq_nbits` 字段（repr(C) 顺序一致）
- `index_factory.h` 路由 `INDEX_FAISS_IVFPQ` → RS，支持 `KNOWHERE_RS_IVFPQ_BYPASS`

**结果（100K×768D IP，m=32，nprobe=64）**：
- Recall@10 = 0.815（IVF-PQ 有损压缩，正常）
- RS c=80: **190.4 QPS**，Native c=80: 191.8 QPS → **parity ✅（<1%）**

**结论**：IVF-PQ Milvus 集成成功，recall 和 QPS 与 native 完全一致。Milvus dispatch ceiling ~191 QPS 主导。

→ 详见 [[benchmarks/authority-numbers]] §IVF-PQ Milvus

---

## 2026-04-08 — IVF-Flat Milvus 集成

**类型**：集成 + bench

**方法**：新建 `ivf_flat_rust_node.cpp` shim（镜像 IVF-SQ8 模式），`CIndexType::IvfFlat=18`，在 `index_factory.h` 中 IVFFLAT/IVFFLAT_CC 路由到 RS，支持 `KNOWHERE_RS_IVFFLAT_BYPASS` env var 对比。

**结果（nprobe=32, 100K×768D IP）**：
- RS c=80: **184.4 QPS**
- Native c=80: 183.2 QPS
- **parity ✅（<1%）**

**结论**：IVF-Flat Milvus 集成成功。上限 ~184 QPS（高于 IVF-SQ8 ~140，因无量化延迟）。Milvus dispatch ceiling 是限制，与 IVF-SQ8 相同结构。Standalone 4.76× 优势不在 Milvus nq=1 模式下体现。

→ 详见 [[benchmarks/authority-numbers]] §IVF-Flat Milvus

---

## 2026-04-08 — IVF-SQ8 IVF_SQ8_CLUSTER_POOL（null result）

**类型**：perf attempt
**Commit**：`5f533c6`

**假设**：`par_iter()` 打全局 rayon pool → 80 CGO 线程 × nprobe=8 = 640 tasks 争 16 线程 → 过订 40×（同 HNSW R6 bug）。添加私有 `IVF_SQ8_CLUSTER_POOL`（镜像 HNSW R7），wrap `install()`。

**结果**：c=1: 38 QPS，c=20: 133 QPS，c=80: **133 QPS（无改善）**

**结论**：cluster-level rayon pool 不是瓶颈。瓶颈是 Milvus dispatch ceiling（~140 QPS），RS 和 native 共享。代码正确保留（防止未来全局池争用），但 IVF-SQ8 @ c=80 天花板由 Milvus 架构决定。IVF-SQ8 无 HNSW 的 nq 批处理机制（nq=1 per FFI call），无法等效复现 HNSW R7 效果。

---

## 2026-04-08 — IVF-SQ8 RS vs Native Milvus 并发对比

**类型**：bench
**数据集**：合成 float32，100K × 768D，IP，nlist=1024

**方法**：`KNOWHERE_RS_IVFSQ8_BYPASS=1` env var 令 index_factory.h 拦截块跳过，IVF-SQ8 回退到 native C++ knowhere。相同 Python 并发脚本，完全可比。

**结果（c=80, nprobe=8）**：RS = 139.5 QPS，Native = 141.1 QPS → **parity ✅（<1% 差异）**

**结论**：RS IVF-SQ8 无额外 per-query overhead。140 QPS 天花板是 Milvus dispatch 天花板，RS 和 native 共享。优化路径：IVF_NQ_POOL（同 HNSW R7），预计 c=80 →500+ QPS。

→ 详见 [[benchmarks/authority-numbers]] §IVF-SQ8 Milvus | `benchmark_results/ivf_sq8_native_milvus_concurrent_2026-04-08.md`

---

## 2026-04-08 — IVF-SQ8 Milvus 并发 QPS 权威数字

**类型**：bench + analysis
**数据集**：合成 float32，100K × 768D，IP，nlist=1024
**机器**：hannsdb-x86

**结果**：
- c=1: nprobe=8→40 QPS，nprobe=32/128→16 QPS（H=25ms 主导）
- c=10: 所有 nprobe 收敛到 134-137 QPS → **dispatch ceiling 确认**
- c=80: **139 QPS 平台期**（HNSW 同条件=1042 QPS）

**结论**：RS IVF-SQ8 计算本身不是瓶颈。瓶颈是 Milvus 每 query 独立 FFI call（无批处理）。HNSW 通过 HNSW_NQ_POOL 将 c=80 的 80 query 批成 1 个 FFI call，IVF-SQ8 缺少等效机制。

**推荐**：实现 IVF_NQ_POOL（参照 HNSW R7），预计 c=80 从 139→500+ QPS（3.5×+）

→ 详见 [[benchmarks/authority-numbers]] §IVF-SQ8 Milvus 并发

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
