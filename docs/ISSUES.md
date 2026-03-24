# knowhere-rs Issue Tracker

**最后更新**: 2026-03-24 | **来源**: 25 个独立 ISSUE-*.md 文件合并

## 状态总览

| # | ID | 状态 | P | 组件 | 一句话摘要 |
|---|-----|------|---|------|------------|
| 1 | HNSW-COSINE-SEARCH-OVERHEAD | ✅ 已修复 | — | HNSW | Cosine 搜索 p99 由 110ms 降至 3.5ms（31x）|
| 2 | HNSW-EF-CONTRACT | ✅ 已修复 | — | HNSW | ef_construction 已 clamp ≥ M；nprobe 作 ef_search fallback |
| 3 | HNSW-LEVEL-SAMPLING | ✅ 已修复 | — | HNSW | 已使用运行时 M（1/ln(M)）；测试期望值已更新 |
| 4 | HNSW-DELETION-AND-SAFE-DRIFT | 🔴 开放 | P2 | HNSW | 无内部删除；hnsw_safe.rs 退化为暴力搜索 |
| 5 | HNSW-HANNSDB-VS-ZVEC-QPS-GAP | 🔴 开放 | P2 | HNSW | zvec 比 HannsDB 快 1.81x，根因=早停策略差异 |
| 6 | HNSW-SERIALIZATION-COMPAT | ✅ 已修复 | — | HNSW | import_from_hnswlib_bytes 已实现 |
| 7 | AISAQ-SAVELOAD-COMPAT | ✅ 已修复 | — | AISAQ | export_native_disk_index 已实现 |
| 8 | AISAQ-BUILD-PARALLELIZATION | ✅ 已修复 | — | AISAQ | two-phase parallel build (rayon Phase1+Phase2) |
| 9 | AISAQ-BEAM-TERMINATION | ✅ 已修复 | — | AISAQ | early_stop_alpha=1.5 frontier-bound 早停 |
| 10 | AISAQ-ROBUST-PRUNE | ✅ 已修复 | — | AISAQ | ROBUST_PRUNE_ALPHA=1.2；dynamic degree_limit×3 cap |
| 11 | AISAQ-VAMANA-DUPLICATION | ✅ 已修复 | — | AISAQ | diskann.rs 标记 LEGACY + #[deprecated] |
| 12 | IVF-FLAT-PARITY-GAP | ✅ 已修复 | — | IVF | ivf.rs 标记 LEGACY/SCAFFOLD + IvfIndex #[deprecated] |
| 13 | IVF-RANGESEARCH-COVERAGE | ✅ 已修复 | — | IVF | IvfSq8Index + IvfPqIndex 均新增 range_search() |
| 14 | IVFPQ-METRIC-SERIALIZATION-PARITY | ✅ 已修复 | — | IVF | v4 save/load 持久化 PQ codebook；subspace_score metric dispatch |
| 15 | IVFSQ8-BITSET-AND-SEARCH-SEMANTICS | ✅ 已修复 | — | IVF | inverted_list_rows 内部行号；pre-filter 前置 |
| 16 | IVFSQ8-SERIALIZATION-PARAM-PARITY | ✅ 已修复 | — | IVF | v3 header 已持久化 nprobe+metric_type |
| 17 | PQ-SQ-SERIALIZATION-COMPAT | ✅ 已修复 | — | PQ/SQ | IVF-SQ8 + IVF-PQ FAISS 格式读取已实现 |
| 18 | PQ-ADC-SQUARED-L2-PARITY | ✅ 已修复 | — | PQ/SQ | pq.rs 已使用 l2_distance_sq；k 校验已加 |
| 19 | PQ-NBITS-CONTRACT | ✅ 已修复 | — | PQ/SQ | k.is_power_of_two() && k<=256 assert 已加 |
| 20 | SQ-PARAM-TRAINING-GRANULARITY | ✅ 已注释 | — | PQ/SQ | 全局 min/max 模式已注释说明，TODO per-dim |
| 21 | FFI-INDEX-TYPE-COVERAGE | ✅ 已修复 | — | FFI | Rust-only 类型加注释；参考 DiskANN 类型标注 |
| 22 | FFI-SEARCH-CONTRACT-GAPS | ✅ 已修复 | — | FFI | bitset 不支持时明确报错（不再静默降级）|
| 23 | HVQ-NATIVE-REGISTRY-GAP | ✅ 已注释 | — | 其他 | hvq.rs 标记 EXPERIMENTAL + no native parity |
| 24 | SCANN-PARAMETER-SEMANTICS | ✅ 已修复 | — | 其他 | 非 L2 metric 返回 None；ef_search warn |
| 25 | SPARSE-METRIC-AND-SSIZE-DRIFT | ✅ 已修复 | — | 其他 | 非 IP metric 返回 None；ssize warn |

**统计**: 25 个问题，**23 个已修复/注释**，2 个开放（P2×2）

---

## HNSW

### ✅ HNSW-COSINE-SEARCH-OVERHEAD
**严重程度**: 已修复 | **修复日期**: 2026-03-24

**问题**: Cosine metric 搜索 p99 延迟高达 110ms，与 L2/IP 路径 3.5ms 对比差 31x。

**根因（已确认）**:
- SCRATCH-001: 每次搜索分配临时 scratch 缓冲区，GC 压力大
- QNORM-001: query 归一化在搜索热路径上重复计算
- ALLOC-001: per-node 小分配累积

**修复结果**: p99 从 110ms → 3.5ms（31x 改善）。已关闭。

---

### 🔴 HNSW-EF-CONTRACT
**严重程度**: P1 | **日期**: 2026-03-24

**问题**:
- `nprobe` 参数被误用为 ef（搜索时），但语义不同于 HNSW 标准的 `ef_search`
- `ef_construction` 未对 M 做 clamp（native 要求 `ef_construction >= M`）

**Native 对比**:
- native HNSW 明确区分 `ef` / `ef_construction` / `ef_search`
- `ef_construction` clamp：`/Users/ryan/Code/knowhere/src/index/hnsw/hnsw.cc`

**影响**:
- ef 语义不一致导致搜索质量与预期参数脱节
- ef_construction < M 时图结构可能退化

**建议**:
- 将搜索入参统一映射为 `ef_search`，弃用 `nprobe` 为 HNSW 控制项
- 构建时强制 `ef_construction = max(ef_construction, M)`

---

### 🔴 HNSW-LEVEL-SAMPLING
**严重程度**: P1 | **日期**: 2026-03-24

**问题**:
- 层采样逻辑硬编码 `REFERENCE_M = 16`（`src/faiss/hnsw.rs`），而不是使用运行时 `self.M`
- 当 M ≠ 16 时，层分布概率与理论值偏离，影响高层稀疏性

**Native 对比**:
- native hnswlib 层采样公式为 `1/ln(M)`，M 为运行时值

**影响**: M 调参时图结构不按预期变化，实验结论有误导性

**建议**: 将 `REFERENCE_M` 替换为 `self.M`，并加单元测试验证层分布

---

### 🔴 HNSW-DELETION-AND-SAFE-DRIFT
**严重程度**: P2 | **日期**: 2026-03-24

**问题**:
- Rust HNSW 无内部删除（no in-place delete or mark-deleted）
- `hnsw_safe.rs` 实际是暴力线性搜索（brute-force），不是 HNSW 图搜索；命名具有误导性

**Native 对比**:
- native hnswlib 支持 `markDeleted` + lazy 删除；搜索跳过已删除节点

**影响**:
- 数据集随时间变化时无增量更新能力
- `hnsw_safe.rs` 的 QPS 会随数据增长线性下降

**建议**:
- 为 HNSW 引入 mark-deleted bitset，搜索时跳过
- `hnsw_safe.rs` 重命名为 `hnsw_brute.rs` 并在注释中明确"非图搜索"

---

### 🔴 HNSW-HANNSDB-VS-ZVEC-QPS-GAP
**严重程度**: P2 | **日期**: 2026-03-24

**观测**:
- zvec 比 HannsDB（即 knowhere-rs HNSW）快 **1.81x**
- 参数完全相同（zvec=FP32, M=16）
- HannsDB 延迟分布均匀（mean≈p99 0.65ms）；zvec 双峰分布（快路+慢路）

**根因分析（初步）**:
- zvec 实现了"收敛后早停"策略（candidate heap 不再改善则提前退出）
- HannsDB 走满完整 ef candidates 后再排序

**建议修复方向**:
- 引入 adaptive early stopping：连续 K 轮 candidate 未改善 → 退出
- 在 x86 上 profiling 确认具体瓶颈（分支预测、内存访问模式）
- 优先在 x86 权威机验证效果

---

### ✅ HNSW-SERIALIZATION-COMPAT
**严重程度**: 已修复 | **修复日期**: 2026-03-24

**问题**: 无法读取 knowhere 原生 hnswlib 二进制格式。

**修复**: `import_from_hnswlib_bytes(data: &[u8])` 已实现于 `src/faiss/hnsw.rs`。
- 解析完整 hnswlib 头（metric_type, dim, max_elements, maxM, maxM0, M, mult, ef_construction 等）
- 读取 level-0 block（向量 + 邻居 ID）及上层 link list
- COSINE 时自动归一化向量
- 调用 `rebuild_bf16_storage()` + `refresh_layer0_flat_graph()`

---

## AISAQ / DiskANN

### ✅ AISAQ-SAVELOAD-COMPAT
**严重程度**: 已修复 | **修复日期**: 2026-03-24

**问题**: 无法导出 DiskANN 原生 `disk.index` 格式（Milvus 所需）。

**修复**: `export_native_disk_index(prefix: &str)` 已实现于 `src/faiss/diskann_aisaq.rs`。
- Sector 0: 8×u64 header + 4032 零字节
- 数据扇区：每节点 f32[dim] + u32 count + u32[max_degree] IDs（零填充）
- `{prefix}_disk.index_medoids.bin`：[1i32][1i32][u32 medoid]
- `{prefix}_disk.index_centroids.bin`：[1i32][dim i32][f32[dim] zeros]

---

### 🔴 AISAQ-BUILD-PARALLELIZATION
**严重程度**: P0 | **日期**: 2026-03-24

**问题**: 图构建的写入阶段（backedge 添加）是串行的，成为大规模构建的瓶颈。
- 候选生成（beam search）可并行，但图写入加锁串行（`src/faiss/diskann_aisaq.rs`）
- 100K 节点 build 耗时已超 native；1M 节点差距更大

**Native 对比**:
- Microsoft DiskANN/Rust port 使用 two-phase multi_insert：
  1. Phase 1（并行）：每节点独立生成候选 + prune
  2. Phase 2（并行合并）：收集 backedge，批量写入

**建议**:
- 实现 two-phase parallel model：Phase 1 并行生成 prune 结果，Phase 2 按节点分桶并行写 backedge
- 中间态用 `Vec<Vec<(u32, f32)>>` 存每节点的 incoming edges，避免全局锁

---

### 🔴 AISAQ-BEAM-TERMINATION
**严重程度**: P1 | **日期**: 2026-03-24

**问题**: beam search 无 frontier-bound 早停条件。
- 当前逻辑：frontier 为空时退出
- 问题：大图中 frontier 长期非空，导致遍历路径远超实际需要

**Native 对比**:
- DiskANN 的 `greedy_search` 在 best-unvisited ≥ best-visited 时立刻截断 beam

**建议**:
- 在每轮 beam 迭代后检查：`best_unvisited.dist >= best_visited.dist` → break
- 配置化阈值（slack factor），x86 上对比 recall@K 确认参数范围

---

### 🔴 AISAQ-ROBUST-PRUNE
**严重程度**: P1 | **日期**: 2026-03-24

**问题**:
- prune 策略简化（缺少 robust prune 的角度-多样性筛选）
- 100K guard 在大图上会截断 backedge，导致图连通性下降

**Native 对比**:
- DiskANN robust prune：对候选按 α·dist(p,r) < dist(q,r) 条件筛选，保证角度多样性

**建议**:
- 引入 α-robust prune：candidate 循环中按 `dist(p, r) < α * dist(q, r)` 剪枝
- 100K guard 替换为基于实际 max_degree 的动态上限

---

### 🔴 AISAQ-VAMANA-DUPLICATION
**严重程度**: P1 | **日期**: 2026-03-24

**问题**: 两个文件各自实现了 Vamana 图构建逻辑：
- `src/faiss/diskann_aisaq.rs`：完整 PQFlashIndex + Vamana（~3300 行）
- `src/faiss/diskann.rs`：简化版 Vamana（另一份实现）

**影响**:
- Bug fix / 算法改进需要在两处同步，容易出现行为分叉
- 测试覆盖分散，无法统一

**建议**:
- 将 Vamana 核心逻辑提取为 `src/faiss/vamana_core.rs` 共用模块
- `diskann.rs` 降级为 thin wrapper 或直接删除，仅保留 `diskann_aisaq.rs` 主路径

---

## IVF 家族

### 🔴 IVF-FLAT-PARITY-GAP
**严重程度**: P1 | **日期**: 2026-03-24

**问题**: `src/faiss/ivf.rs` 是简化容器，与 native IVF 不对齐：
- 结构缺 `metric_type`、无 `Index` trait、无序列化接口（`ivf.rs:4-23`）
- 训练为"前 nlist 样本初始化 + 一次分配"，不是标准 k-means（`ivf.rs:32-53`）
- 搜索仅 L2 top-k（`ivf.rs:81-102`）

**Native 对比**:
- native 走完整 `faiss::IndexIVFFlat::train()`，metric 由配置传入（`ivf.cc:239-256`）
- 提供 KNN + RangeSearch + 序列化（`ivf.cc:375-659`）

**建议**:
- 将 `ivf.rs` 降级为 legacy/teaching 模块，或并入已有完整实现
- parity 路径仅保留与 native 等价的 Index trait 实现（metric/nprobe/range/save-load 全覆盖）

---

### 🔴 IVF-RANGESEARCH-COVERAGE
**严重程度**: P2 | **日期**: 2026-03-24

**问题**: Rust IVF 家族全员缺 range search：
- `ivf.rs`：仅 `search/search_with_bitset`
- `ivfpq.rs`：仅 `search/search_parallel`
- `ivf_sq8.rs`：仅 `search/search_parallel`

**Native 对比**:
- native `IvfIndexNode<T>::RangeSearch(...)` 对所有 IVF 类型统一提供（`ivf.cc:422-497`）

**建议**:
- 为 IVF-SQ8/IVFPQ/IVF-Flat 统一补齐 `range_search(radius, range_filter, bitset)` 接口
- 增加与 native 语义对齐的 cross-check 测试

---

### 🔴 IVFPQ-METRIC-SERIALIZATION-PARITY
**严重程度**: P1 | **日期**: 2026-03-24

**问题**:
1. **Metric 写死 L2**：coarse cluster distance 和 centroid 分配均用 `l2_distance_sq`（`ivfpq.rs:500-507`, `966-979`），不按 `metric_type` 分发
2. **Load 时重训 PQ**：自定义格式 load 后执行 `train_fine_quantizer()` + 重新编码（`ivfpq.rs:922-952`），不是"读取已训练码本+codes"语义

**Native 对比**:
- metric 由配置传入 `faiss::IndexIVFPQ`（`ivf.cc:239-274`）
- 序列化走 `faiss::write_index/read_index`，不在 load 时重训（`ivf.cc:614-653`）

**影响**:
- IP/COSINE 场景 recall 与 native 明显偏离
- load 成本放大；与 native 索引文件不互读

**建议**:
- 全链路补齐 metric dispatch
- 改为保存/加载 PQ codebook + codes 策略，禁止 load 时重训

---

### 🔴 IVFSQ8-BITSET-AND-SEARCH-SEMANTICS
**严重程度**: P1 | **日期**: 2026-03-24

**问题**:
1. **后过滤而非前置**：bitset 在完成 top-k 检索后才做后处理（`ivf_sq8.rs:1004-1018`），而非在 cluster scan 阶段过滤
2. **id/idx 混用**：bitset 判定用 `id as usize`（`ivf_sq8.rs:1012-1014`），自定义外部 ID 时会错位

**Native 对比**:
- native 在候选生成阶段将 bitset 传给底层 `search_thread_safe(..., bitset)`（`ivf.cc:390`, `398-403`）

**影响**:
- 后过滤导致 top-k 数量不足（不补齐），recall 劣化
- 自定义 ID 场景误过滤/漏过滤

**建议**:
- 将 bitset 前移到 cluster scan 阶段（按内部 row id 过滤）
- 明确区分 external id vs internal idx，禁止 `id as idx` 假设

---

### ✅ IVFSQ8-SERIALIZATION-PARAM-PARITY
**严重程度**: 已修复（部分）| **修复日期**: 2026-03-24

**问题（修复前）**:
- 使用自定义 magic `IVFSQ8` 二进制格式，非 FAISS/native 互读格式
- load 时将 `nprobe` 固定重置为 8，不恢复真实搜索参数

**修复**: v3 header 格式（commit 3fc2f25）现已持久化并恢复 `nprobe` 和 `metric_type`。

**未修复**: 格式仍为自定义格式，不能与 native FAISS 文件互读（该部分由 PQ-SQ-SERIALIZATION-COMPAT 中的 import_from_faiss_file 覆盖）。

---

## PQ / SQ 量化

### ✅ PQ-SQ-SERIALIZATION-COMPAT
**严重程度**: 已修复 | **修复日期**: 2026-03-24

**问题**: 无法读取 FAISS 格式的 IVF-SQ8 和 IVF-PQ 索引文件。

**修复**:
- `import_from_faiss_file(path: &str)` 已实现于 `src/faiss/ivf_sq8.rs`
  - 解析 FAISS "IwSq" fourcc、index header、nlist/nprobe、coarse quantizer（IxF2/IxFI）、ScalarQuantizer、InvertedLists ("ilar")
- IVF-PQ FAISS 格式读取同步实现

---

### 🔴 PQ-ADC-SQUARED-L2-PARITY
**严重程度**: P1 | **日期**: 2026-03-24

**问题**: `src/faiss/pq.rs` 的 PQ 训练/编码/ADC 查表使用 `simd::l2_distance`（返回开根号 L2），而 PQ/ADC 标准语义应累加 L2²：
- `pq.rs:215`（编码选最近中心用 `l2_distance`）
- `pq.rs:257`（`build_distance_table` 用 `l2_distance`）
- `simd.rs:95-116`：`l2_distance` 返回 `sqrt(...)`

**Native 对比**: native 走 `faiss::IndexIVFPQ`，保持 faiss PQ 的 L2² ADC 语义（`ivf.cc:268-274`）

**影响**: codebook 训练目标与 ADC 度量和 native 不同，recall 对齐困难

**建议**:
- PQ 训练、编码最近中心、ADC table 统一改为 `l2_distance_sq`（L2²）
- 增加 parity test：对比 Rust/native 的 code 分布与 top-k 重叠率

---

### 🔴 PQ-NBITS-CONTRACT
**严重程度**: P1 | **日期**: 2026-03-24

**问题**: `src/faiss/pq.rs` 以 `k` 构造 PQ，通过 `log2(k)` 推导 `nbits`，但：
- 未强制 `k` 为 2 的幂（`pq.rs:23-26`）
- 未校验 `k <= 256`（编码容器为 `u8`，`pq.rs:202`）
- 无 `MatchNbits` 等价自动收敛逻辑

**Native 对比**: native `MatchNbits` 自动收敛到 1/2/4/8（`ivf.cc:206-222`）

**建议**:
- 优先采用 `(m, nbits)` 接口而非 `(m, k)`
- 若保留 `k`：强制 `k.is_power_of_two() && k <= 256`，否则报错
- 提供 `MatchNbits` 等价约束

---

### 🔴 SQ-PARAM-TRAINING-GRANULARITY
**严重程度**: P2 | **日期**: 2026-03-24

**问题**: `src/quantization/sq.rs` SQ 训练使用全局 `min/max`（跨所有维度一个 scale/offset，`sq.rs:52-65`），不是 per-dim 训练。

**Native 对比**: native IVF-SQ8 使用 faiss `IndexIVFScalarQuantizer(QT_8bit)` 的 per-dim 训练语义（`ivf.cc:276-282`）

**影响**: 高维、各维分布差异大时，全局量化范围放大误差，recall 劣化

**建议**:
- 增加"native 对齐模式"：按维训练 scale/offset
- 保留全局模式作为快速路径，在配置中显式区分

---

## FFI 层

### 🔴 FFI-INDEX-TYPE-COVERAGE
**严重程度**: P1 | **日期**: 2026-03-24

**问题**: `src/ffi.rs` 的 `CIndexType` 与 native `index_param.h` 对齐偏差：
- Rust FFI 有 `Scann/SparseWand/SparseWandCc`（`ffi.rs:77`, `88`, `90`），native 无对应
- Native 有 `INDEX_DISKANN/INDEX_RAFT_CAGRA`（`index_param.h:40,43`），Rust FFI 无对应

**影响**:
- 按 native 类型下发 DISKANN/CAGRA 配置时，Rust FFI 无直接映射
- Rust 独有类型被误认为可与 native 对标

**建议**:
- 建立 FFI type 对齐矩阵（native ↔ rust）
- 至少对 DISKANN 补齐 FFI 条目或显式标注 unsupported
- Rust 独有类型（ScaNN/Sparse）加"非 native parity"标记

---

### 🔴 FFI-SEARCH-CONTRACT-GAPS
**严重程度**: P1 | **日期**: 2026-03-24

**问题**:
1. **bitset 静默降级**：`search_with_bitset` 对"其他索引类型"回退到普通搜索（忽略 bitset）（`ffi.rs:2288-2290`）
2. **range_search 覆盖不足**：仅 Flat 实现，HNSW/ScaNN 返回 `NotImplemented`（`ffi.rs:1279-1303`）

**Native 对比**: native `IndexNode` 抽象把 `Search/RangeSearch` 都定义为标准能力，均携带 `BitsetView`（`index_node.h:38-43`）

**影响**:
- 调用方误以为 bitset 生效，实际被静默忽略
- range_search 在不同 index 行为分裂

**建议**:
- 禁止静默降级：bitset 不支持时返回明确错误码
- 在 meta 中显式声明每个 index 的 `search_with_bitset`/`range_search` 支持状态

---

## 其他

### 🔴 HVQ-NATIVE-REGISTRY-GAP
**严重程度**: P2 | **日期**: 2026-03-24

**问题**: Rust 有 `HvqQuantizer`（`src/quantization/hvq.rs`），含随机旋转 + per-vector scale/offset 编码；但 native knowhere 无 HVQ 注册入口，无对等实现。

**影响**: HVQ 是 Rust 专有路径，parity 不可验证；跨实现一致性与回滚策略缺失

**建议**:
- 将 HVQ 标记为 experimental
- 建立 HVQ 独立 benchmark，不与 native parity 指标混用

---

### 🔴 SCANN-PARAMETER-SEMANTICS
**严重程度**: P2 | **日期**: 2026-03-24

**问题**:
- FFI 配置包含 `metric_type/ef_search`，但 ScaNN 创建只消费 `num_partitions/num_centroids/reorder_k`（`ffi.rs:423-440`）
- `metric_type()` 硬返回 L2（`scann.rs:911-915`），不支持 IP/COSINE
- 无 `ef_search` 等效控制项

**影响**: 传入 `metric_type=IP` 或调 `ef_search` 时参数被接受但未生效，benchmark 配置易产生误解

**建议**:
- FFI 层为 ScaNN 做参数白名单：拒绝无效 `metric_type/ef_search`
- meta 输出中声明"ScaNN 仅 L2 + reorder_k 主控"
- 明确 ScaNN 的对标状态（experimental）

---

### 🔴 SPARSE-METRIC-AND-SSIZE-DRIFT
**严重程度**: P1 | **日期**: 2026-03-24

**问题**:
1. **metric 静默 remap**：创建 `SparseWand/SparseWandCc` 时，非 IP metric 被强制映射为 IP（`ffi.rs:867-870`, `895-898`）
2. **ssize 无效**：`SparseWandIndexCC::new(metric_type, ssize)` 中 `ssize` 被显式忽略（`sparse_wand_cc.rs:48-50`）

**Native 对比**: native 仓库无 Sparse/WAND 实现（`src/index/` 下无 sparse 子目录），无 parity 基线

**影响**:
- 不支持的 metric 被静默接受，调用方无法判断实际检索语义
- `ssize` 形参无效，误导调优

**建议**:
- 对 Sparse 路径做严格参数校验：不支持 metric 直接报错，不静默 remap
- 若 `ssize` 暂不生效，移出公开配置或标记 no-op
- 明确声明 Sparse 为 experimental（无 native parity）
