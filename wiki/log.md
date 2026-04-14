# Log — hanns

append-only 时间线。新条目加在顶部。

---

## 2026-04-14 — HannsDB vs zvec：storage/query parity 收尾改为“状态诚实性优先”

**类型**：decision / refactor
**仓库**：HannsDB

**结果摘要**：
- 当前 dirty slice 没有继续扩新 parity 面，而是先收敛 storage/query 主线：`db.rs` 存储职责拆分、显式校验、ANN 状态回归测试。
- 新增 `storage/{paths,primary_keys,recovery,segment_io,wal}.rs`，并把共享结构拆到 `db_types.rs` / `query/hits.rs`。
- 明确验证 `index_completeness` / `ann_ready` 的三段状态：optimize 后成立、reopen 后保持、subsequent write 后清空。
- daemon 层也补齐 router rebuild / write-after-optimize 的同类状态验证。

**关键判断**：
- 这轮对 zvec 缩小的是真实 runtime 成熟度差距的一部分，而不是新的 surface 名义 parity。
- zvec 的整体 versioned segment/runtime 体系仍然更成熟；HannsDB 这轮主要是把“对外宣称的状态”做成事实。

**验证**：
- `cargo test -p hannsdb-core --test collection_api --test wal_recovery -- --nocapture`
- `cargo test -p hannsdb-daemon --test http_smoke -- --nocapture`
- `bash scripts/run_zvec_parity_smoke.sh`
- parity smoke 通过：Rust parity + lifecycle + compaction + collection_api + wal_recovery + segment_storage + daemon + Python（168 passed / 4 skipped）

→ 详见 [[decisions/hannsdb-zvec-storage-query-parity]]。

---

## 2026-04-10 — Milvus Cohere-1M：PCA / SQ / IVF-SQ8 benchmark 更新

**类型**：bench
**机器**：hannsdb-x86
**数据集**：Cohere-MEDIUM-1M，768D，COSINE，k=100

**结果摘要**：
- HNSW baseline：load 6186.8944s，QPS 166.8374，recall 0.9352，p99 0.0108s
- HNSW-PCA-USQ：load 8088.0279s，QPS 173.9785，recall 0.9445，p99 0.0103s
- HNSW-SQ（SQ8 + FP32 refine）：load 6101.9038s，QPS 198.6828，recall 0.9474，p99 0.0099s
- IVF-SQ8（nlist=1024, nprobe=64）：load 984.2313s，QPS 8.0383，recall 1.0000，p99 0.6117s

**失败项**：
- DiskANN-PCA-USQ：第一次用 `search_list=100, k=100` 运行，load 已完成（insert 159.0386s + optimize 1447.5195s），但搜索失败。
- 根因：Milvus QueryNode 参数校验触发：`search_list_size(100) should be larger than k(100)`。
- 后续 VectorDBBench 在空结果上做 percentile，又触发 `IndexError: index -1 is out of bounds for axis 0 with size 0`。

**补充复测（2026-04-10 晚）**：
- 用合法参数 `search_list=128, k=100` 重新跑通 DiskANN-PCA-USQ。
- 结果：load 239.1671s，QPS 8.1176，recall 1.0000，p99 1.2366s。
- 结论：问题确实是参数非法，不是 DiskANN-PCA-USQ 主流程完全不可用；但即使跑通，当前性能也明显弱于 HNSW-SQ / HNSW-PCA-USQ。

**结论**：
- 在这批 1M Milvus 集成 benchmark 中，**HNSW-SQ 当前综合表现最好**（QPS / recall / p99 都优于 baseline 与 HNSW-PCA-USQ）。
- HNSW-PCA-USQ 没有退化，QPS 比 baseline 略高（173.98 vs 166.84），recall 也更高（0.9445 vs 0.9352），但 build/optimize 时间更长。
- IVF-SQ8 的 build/load 明显更快，但搜索吞吐极低，不适合当前目标 workload。
- DiskANN-PCA-USQ 必须保证 `search_list > k`；修正后可跑通，但当前性能不具备竞争力。

→ 详见 [[benchmarks/authority-numbers]] 新增 2026-04-10 小节。

---

## 2026-04-10 — IndexWrapper Option-soup 重构（IndexKind 枚举）

**类型**：refactor
**文件**：`src/ffi.rs`

将 `IndexWrapper` 的 20 个 `Option<ConcreteType>` 字段（永远只有 1 个 `Some`）替换为单一 `enum IndexKind` 枚举变体。

**变更要点**：
- `new()` 工厂函数从 ~890 行缩减至 ~250 行（消除每个变体需要设 19 个 `None` 的样板）
- 所有方法 dispatch 从 `if let Some(ref idx) = self.flat { ... } else if ...` 改为 `match &self.kind { IndexKind::Flat(idx) => ... }`
- `knowhere_search_with_bitset` 内联 ~150 行 dispatch 提取为 `IndexWrapper::search_with_bitset()` 方法
- `search()` 新增 `query_dim` 参数，修复稀疏索引处理大维度 query 时的 bug
- `SparseWandCc` 在 `search()` 中补充了遗漏的分支

**净效果**：`src/ffi.rs` 7613 → 5846 行（减少 ~1100 行），663 个 lib 测试全部通过。

---

## 2026-04-10 — Cohere 1M HNSW：Hanns vs Lance 对比

**类型**：bench
**机器**：ecs-hk-1b (96 vCPU, 493GB RAM, Linux x86_64)
**数据集**：Cohere 1M × 1024d (1,000,000 vectors), L2 metric
**方法**：Hanns HNSW (`add_parallel`, 64T) vs Lance HNSW (`index_vectors`, rayon multi-thread)，ef sweep [50,100,200,400,800]

**Build**：Hanns 129.63s vs Lance **96.45s** (Lance 快 26%)

**Search（K=10, 100 queries）**：

| ef | Hanns recall | Hanns QPS | Lance recall | Lance QPS | Ratio |
|----|-------------|-----------|-------------|-----------|-------|
| 50 | 0.9950 | 2331 | 0.9960 | 1473 | 0.63 |
| 100 | 0.9950 | 1433 | 0.9970 | 897 | 0.63 |
| 200 | 0.9950 | 794 | 0.9970 | 483 | 0.61 |
| 400 | 0.9960 | 443 | 0.9990 | 261 | 0.59 |
| 800 | 0.9960 | 245 | 1.0000 | 140 | 0.57 |

**Search ratio (Lance/Hanns)**：0.61 geometric mean → **Hanns search ~1.6× 更快**

**关键发现**：
- 两者 recall 均极高（≥0.995），Lance 略高（ef≥200 时 0.997-1.000）
- Hanns search QPS 在所有 ef 点均显著领先（1.56-1.75×）
- Lance build 快 26%（rayon 并行 vs Hanns parallel batch 串行瓶颈 18%）
- Hanns search 优势来自：SIMD 距离计算优化 + 图遍历更紧凑（M=16 时 Hanns 搜索邻居数更少）

→ 详见 [[benchmarks/authority-numbers]] §HNSW vs Lance

---

## 2026-04-10 — HannsDB VectorDBBench Standalone Benchmark

**类型**：bench
**机器**：knowhere-x86-hk-proxy (94.74.108.167) + 本地 MacBook (ARM64)
**数据集**：OpenAI 1536D 50K, cosine

**x86 结果（HNSW, M=16, ef_construction=64, ef_search=32, k=100）**：
- Load: 148.0s（insert 60.2s + optimize 87.9s）
- p99 latency: 1.8ms
- p95 latency: 1.7ms
- Recall@100: 0.9756
- NDCG@100: 0.9801

**本地 ARM64 结果（同参数, k=100）**：
- Load: 215.2s
- p99 latency: 1.4ms
- p95 latency: 1.0ms
- Recall@100: 0.9756
- NDCG@100: 0.9801

**本地 ARM64（k=10）**：
- Load: 218.8s
- p99 latency: 0.5ms
- p95 latency: 0.3ms
- Recall@10: 0.9441

**HannsDB 内部 opt bench（x86, knowhere-backend, 50K/1536D/cosine）**：
- create=0ms, insert=32504ms, optimize=18929ms, search=0ms, total=51435ms

**HannsDB vs Zvec VDBB 对比（同机器, 同参数）**：

| 指标 | HannsDB | Zvec | 比值 |
|------|---------|------|------|
| Load | 148.0s | 13.8s | Zvec 10.7× 快 |
| p99 | 1.8ms | 2.0ms | 持平 |
| p95 | 1.7ms | 1.3ms | 持平 |
| Recall@100 | **0.9756** | 0.9286 | HannsDB +5.1% |
| NDCG@100 | **0.9801** | 0.941 | HannsDB +4.2% |

**关键发现**：
- Zvec load 极快（C++ 原生 + Arrow 列存），但 recall 明显低
- HannsDB 同参数下 ANN 质量（recall/NDCG）显著更高
- Search latency 两者持平

→ 详见 [[machines/knowhere-x86-hk-proxy]]

---

## 2026-04-09 — 代码库清理

**类型**：chore

**操作**：
1. 删除 14 个零导入模块 + 7 个孤儿文件（共 2,228 行死代码）：once_cell, skiplist, ring, prealloc, arena, atomic_utils, layout, lru_cache, stats, federation, bloom, storage/, annoy, lazy_index, raw, hnsw_safe, ivf_flat_search_v25, faiss_index
2. 提取测试公共工具到 `tests/common/mod.rs`：`generate_vectors`（12→1）、`l2_distance_squared`（27→1）、`compute_ground_truth`（14→1）
3. 修复 2 个 diskann disk path 测试（断言与 NoPQ materialize 行为不匹配）
4. `git filter-repo` 清除历史中的 `target/`（.git 273M→6.4M，−97.7%）
5. 整理文档：删除 102 个 superpowers 会话产物，归档 11 个历史 docs，清空 memory/

**验证**：663 tests passed / 0 failed

---

## 2026-04-09 — 项目目录结构整理

**类型**：chore

**操作**：
1. 根目录脚本移入 `scripts/`：`add_scann.py`, `test_*.py`, `build.sh`, `init.sh`, `init.ps1`, `compare_cpp_rust_simd.sh`
2. `assets/benchmarks/`（5 个 PNG）合并进 `benchmark_results/`，`assets/` 删除
3. `benchmark_results/` 中 57 个历史 .json 归档到 `archive/`，顶层只保留 4 个 .md 报告 + 5 个 PNG
4. `feature-list.json`（75K）, `performance_baseline.json` 移入 `benchmark_results/archive/`
5. 空目录 `memory/` 删除

**根目录文件数**：19 → 7（AGENTS.md, CLAUDE.md, README.md, Cargo.toml, Cargo.lock, build.rs, CMakeLists.txt）

---

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
- `index_factory.h` 路由 `INDEX_FAISS_IVFPQ` → RS，支持 `HANNS_IVFPQ_BYPASS`

**结果（100K×768D IP，m=32，nprobe=64）**：
- Recall@10 = 0.815（IVF-PQ 有损压缩，正常）
- RS c=80: **190.4 QPS**，Native c=80: 191.8 QPS → **parity ✅（<1%）**

**结论**：IVF-PQ Milvus 集成成功，recall 和 QPS 与 native 完全一致。Milvus dispatch ceiling ~191 QPS 主导。

→ 详见 [[benchmarks/authority-numbers]] §IVF-PQ Milvus

---

## 2026-04-08 — IVF-Flat Milvus 集成

**类型**：集成 + bench

**方法**：新建 `ivf_flat_rust_node.cpp` shim（镜像 IVF-SQ8 模式），`CIndexType::IvfFlat=18`，在 `index_factory.h` 中 IVFFLAT/IVFFLAT_CC 路由到 RS，支持 `HANNS_IVFFLAT_BYPASS` env var 对比。

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

**方法**：`HANNS_IVFSQ8_BYPASS=1` env var 令 index_factory.h 拦截块跳过，IVF-SQ8 回退到 native C++ knowhere。相同 Python 并发脚本，完全可比。

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
