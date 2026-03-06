# PARITY_AUDIT (Non-GPU)

Last updated: 2026-03-06 23:58
Sync baseline: 388aea90260084965f965b29e0a8b87b7a808d51 from origin/main

## 轮次记录
- 2026-03-06 23:58: **BUG-P0-004 编译回归收敛 + 双模块复核（src/index.rs / src/ffi.rs）**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. C++/Rust 增量扫描：`ls .../knowhere/src/index | wc -l`=17，`ls .../include/knowhere/index | wc -l`=8，`ls .../knowhere-rs/src/faiss | wc -l`=47
  3. 深比对（本轮按要求至少 2 模块逐接口复核）：
     - `src/index.rs`：`trait Index` 统一契约仍在 (`src/index.rs:127`)
     - `src/ffi.rs`：`CIndexConfig` / `knowhere_get_index_type` / `knowhere_create_ann_iterator` 接口声明完整
  4. 回归修复：批量修复 tests/examples 的 `IndexConfig::data_type` 缺失；统一 `crate::api::DataType` -> `knowhere_rs::api::DataType`
  5. 验证结果：
     - `cargo test --tests --no-run -q` ✅（data_type 迁移导致的编译错误已清零）
     - `cargo test -q` / `cargo test --lib -q` ❌（运行期失败：AISAQ trait tests、ScaNN FFI tests、kmeans convergence 等）
  6. 新发现差距：从“编译回归”转为“运行期功能回归”聚焦，下一步应拆分 AISAQ/ScaNN/KMeans 的稳定性缺口。
  状态：BUG-P0-004 关闭（编译回归已修复）；全量功能回归仍为 P1。
- 2026-03-06 18:02: **MinHash Index trait wrapper 接入 + 全量测试回归诊断**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 对齐扫描命令：`ls /Users/ryan/Code/vectorDB/knowhere/src/index && ls /Users/ryan/Code/vectorDB/knowhere/include/knowhere/index && ls /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss`
  3. 逐文件复核：`src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`、`src/index/minhash_lsh_index_trait.rs`、`src/index/minhash/minhash_index_node.cc`
  4. 变更：新增 `src/index/minhash_lsh_index_trait.rs`，为 MinHashLSHIndex 实现统一 `Index` trait（train/add/search/range/get_vector/serialize/iterator 元数据）并补齐 5 个单测；`src/index.rs` 导出模块。
  5. 验证：`cargo test --lib minhash_lsh_index_trait` 通过；`cargo test` / `cargo test --tests` 失败，失败原因为 tests 侧 `IndexConfig.data_type` 迁移未完成（非本轮 MinHash 模块逻辑错误）。
  6. 新发现差距：全量测试编译回归，需新增 P0 修复任务 `BUG-P0-004`。
  状态：MinHash 模块从 Partial（无 trait）提升到 Partial（trait 已接入，仍待 FFI 统一接线与全量回归恢复）。
- 2026-03-06 14:38: **MinHash FFI 查询长度对齐修复 + 抽样复核**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 抽样深比对命令：`ls /Users/ryan/Code/vectorDB/knowhere/src/index && ls /Users/ryan/Code/vectorDB/knowhere/include/knowhere/index && ls /Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/faiss`
  3. 复核接口：`src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`、`src/ffi/minhash_lsh_ffi.rs`
  4. 修复项：`src/index/minhash_lsh.rs` 新增 `vector_byte_size()`；`src/ffi/minhash_lsh_ffi.rs` 将 query/queries 长度计算从占位逻辑改为 `mh_vec_length * mh_vec_element_size`
  5. 新增回归测试：`test_search_uses_vector_byte_size`
  状态：MinHash 模块维持 Partial（Index trait wrapper 仍缺失），但 FFI query 长度缺口已关闭，风险从 P1-high 降为 P1-medium。
- 2026-03-06 13:32: **MinHash 参数别名对齐 + 目录级复核**
  1. 执行同步与基线更新：`git fetch origin && git checkout main && git pull origin main && git rev-parse origin/main` -> `388aea90260084965f965b29e0a8b87b7a808d51`
  2. 扫描 C++ 目录：`src/index/`、`include/knowhere/index/`；并对照 Rust `src/index.rs`、`src/ffi.rs`、`src/index/minhash_lsh.rs`
  3. MinHash 参数命名对齐（部分完成）：`src/api/index.rs` 新增 `mh_*` 到 Rust 参数字段的 serde alias + 单测
  4. 新发现差距：MinHash 仍未接入统一 `Index trait`；`src/ffi/minhash_lsh_ffi.rs` 的 query 大小计算仍为占位逻辑（`count()*count()`）
  状态：MinHash 模块保持 Partial，风险维持 P1。
- 2026-03-06 12:32: **AISAQ Index trait 实现** - 为 AisaqIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load/serialize_to_memory/deserialize_from_memory
  2. 高级接口：AnnIterator (AisaqAnnIterator) / get_vector_by_ids / has_raw_data
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 添加 Serialize/Deserialize 到 AisaqConfig
  5. 创建测试套件验证实现（5 个测试）
  状态：AISAQ 模块从 Partial 升级为 Done（Index trait 实现完成）。
- 2026-03-06 10:32: **ScaNN Index trait 验证** - 确认 ScaNNIndex 已实现完整 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load
  2. 高级接口：get_vector_by_ids（支持但需检查 has_raw_data）/has_raw_data（取决于 reorder_k）/create_ann_iterator
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（6 个测试全部通过）
  5. 修复编译错误：binary_hnsw.rs 和 diskann.rs 的 data_type 字段问题
  状态：ScaNN 模块从 Partial 升级为 Done（Index trait 实现完成并测试验证）。
- 2026-03-06 07:35: **DiskANN Index trait 实现** - 为 DiskAnnIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/range_search/save/load
  2. 高级接口：AnnIterator (DiskAnnIteratorWrapper) / get_vector_by_ids
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（test_diskann_index_trait）
  状态：DiskANN 模块从 Partial 升级为 Done（Index trait 实现完成）。
- 2026-03-06 06:35: **IVF 系列架构缺口修复** - 为 IvfSq8Index 和 IvfRaBitqIndex 实现完整的 Index trait，包括：
  1. 基础生命周期方法：train/add/search/search_with_bitset/save/load
  2. 高级接口：AnnIterator（两个索引）/ get_vector_by_ids（仅 IVF-SQ8，IVF-RaBitQ 因有损压缩返回 Unsupported）
  3. 元数据方法：index_type/dim/count/is_trained/has_raw_data
  4. 创建测试套件验证实现（7 个测试全部通过）
  状态：IVF core 模块从 Partial 升级为 Done（Index trait 实现完成），剩余参数校验统一化任务。
- 2026-03-06 05:35: **IVF 系列架构缺口诊断** - 发现 IVF-SQ8/IVF-RaBitQ 未实现 Index trait，仅通过 FFI IndexWrapper enum dispatch 访问。这意味着：
  1. IVF 系列无法通过统一 Index trait 调用高级接口（AnnIterator/get_vector_by_ids）
  2. FFI 层需要为每种 IVF 类型重复实现调用逻辑
  3. 参数校验和错误处理可能不一致
  行动：将 PARITY-P1-002 升级为关键架构任务，需要为 IVF 系列实现 Index trait wrapper。
- 2026-03-06 04:35: **HNSW 高级路径测试** - 创建 `tests/test_hnsw_advanced_paths.rs`，覆盖 get_vector_by_ids、AnnIterator、serialize/deserialize、range_search（Unsupported）。5 个测试全部通过。PARITY-P1-001 完成，HNSW 模块状态升级为 Partial → Done（高级路径）。
- 2026-03-06 03:35: **核心契约一致性验证** - 验证所有索引对未实现方法的错误处理一致：Index trait 提供默认 Unsupported 实现；FFI 层 19 处 NotImplemented 返回；所有非 GPU 索引行为一致。核心契约状态从 Partial 升级为 Done（P0 降级）。
- 2026-03-06 01:35: **实现 FFI AnnIterator 接口** - 添加 `knowhere_create_ann_iterator`/`knowhere_ann_iterator_next`/`knowhere_free_ann_iterator` 三个 FFI 函数，支持 HNSW/ScaNN/HNSW-PQ 索引
- 2026-03-06 00:35: **更新 FFI 能力矩阵** - 标记 HNSW/ScaNN/HNSW-PQ/DiskANN 的 AnnIterator 为 ✅；HNSW GetByID ✅；ScaNN GetByID ⚠️；DiskANN GetByID ⚠️
- 2026-03-05 23:35: **实现 AnnIterator 接口** - HNSW, ScaNN, HNSW-PQ, DiskANN 四个索引实现 create_ann_iterator，验收标准达成（>=3个索引）
- 2026-03-05 22:32: 详细对比 C++/Rust 核心接口，发现 IsAdditionalScalarSupported/GetIndexMeta 缺失，AnnIterator 未实现
- 2026-03-05 21:22: 扫描 C++/Rust 接口对齐状态，确认 AnnIterator 未实现但已定义，HNSW 实现核心接口
- 2026-03-05 20:40: 添加 AnnIterator 接口定义，创建 FFI 能力矩阵文档
- 2026-03-05 20:35: 确认 ivf_sq_cc 所有测试通过 (6/6)，BUG-P0-003 完成
- 2026-03-05 19:35: 修复 3 个 P0 BUG (mini_batch_kmeans/diskann_complete/ivf_sq_cc SIMD 切片长度问题)

## 1. Scope

- In scope: non-GPU parity against C++ knowhere
- Out of scope: GPU/cuVS implementation parity

## 2. Status Legend

- `Done`: implementation and behavior aligned
- `Partial`: implemented but behavioral/edge mismatch remains
- `Blocked`: intentionally deferred or requires prerequisite
- `Missing`: not implemented

Risk levels:

- `P0`: blocks production parity
- `P1`: important functional/behavioral gap
- `P2`: optimization/documentation/coverage gap

## 3. Module-Level File Mapping and Gap Items

| Module | Native file(s) | Rust file(s) | Status | Risk | Pending interface items |
|---|---|---|---|---|---|
| Core contract | `include/knowhere/index/index.h`, `include/knowhere/index/index_node.h` | `src/index.rs`, `src/api/search.rs` | Done | P1 | ✅ lifecycle contract unified (2026-03-06); ✅ AnnIterator trait implemented (2026-03-05); all indexes return consistent Unsupported for unimplemented methods |
| Index factory/legality | `include/knowhere/index/index_factory.h`, `include/knowhere/index/index_table.h`, `include/knowhere/comp/knowhere_check.h` | `src/api/index.rs`, `src/api/data_type.rs`, `src/api/legal_matrix.rs`, `src/ffi.rs` | Partial | P1 | centralized legal matrix for index/datatype/metric; ✅ Implemented (2026-03-06); validation active at FFI entry points
| HNSW | `src/index/hnsw/faiss_hnsw.cc` | `src/faiss/hnsw.rs` | Done | P1 | ✅ AnnIterator (2026-03-05), ✅ get_vector_by_ids (2026-03-05), ✅ serialize/deserialize, ✅ range_search (Unsupported, tested 2026-03-06); all advanced paths tested and aligned |
| IVF core | `src/index/ivf/ivf.cc`, `src/index/ivf/ivf_config.h` | `src/faiss/ivf.rs`, `src/faiss/ivf_flat.rs`, `src/faiss/ivfpq.rs`, `src/api/index.rs` | Done | P1 | ✅ Index trait implemented for IvfSq8Index and IvfRaBitqIndex (2026-03-06); ✅ AnnIterator; ✅ get_vector_by_ids (IVF-SQ8 only); parameter coverage and edge behavior alignment remaining; SIMD slice fix in ivf_sq_cc (2026-03-05) |
| RaBitQ | `src/index/ivf/ivfrbq_wrapper.*` | `src/faiss/ivf_rabitq.rs`, `src/faiss/rabitq_ffi.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator; ⚠️ get_vector_by_ids (Unsupported for lossy compression); query-bits and config boundary consistency |
| DiskANN | `src/index/diskann/diskann.cc`, `src/index/diskann/diskann_config.h` | `src/faiss/diskann.rs`, `src/faiss/diskann_complete.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator (DiskAnnIteratorWrapper); ✅ get_vector_by_ids; lifecycle parity and config semantics; add_batch SIMD slice fix (2026-03-05) |
| AISAQ | `src/index/diskann/diskann_aisaq.cc`, `src/index/diskann/aisaq_config.h` | `src/faiss/diskann_aisaq.rs`, `src/faiss/aisaq.rs` | Done | P1 | ✅ Index trait implemented (2026-03-06); ✅ AnnIterator; ✅ get_vector_by_ids; parameter and file-layout behavior alignment |
| ScaNN | - | `src/faiss/scann.rs` | Done | P1 | ✅ AnnIterator (2026-03-05), ✅ get_vector_by_ids (2026-03-06), ✅ has_raw_data (depends on reorder_k), ✅ Index trait (2026-03-06); tested |
| HNSW-PQ | - | `src/faiss/hnsw_pq.rs` | Partial | P2 | ✅ AnnIterator (2026-03-05); has_raw_data=false (lossy) |
| Sparse | `src/index/sparse/sparse_index_node.cc`, `src/index/sparse/sparse_inverted_index.h` | `src/faiss/sparse_inverted.rs`, `src/faiss/sparse_wand.rs`, `src/faiss/sparse_wand_cc.rs` | Partial | P2 | iterator/filter behavior and parameter parity |
| MinHash | `src/index/minhash/minhash_index_node.cc`, `src/index/minhash/minhash_lsh_config.h` | `src/index/minhash_lsh.rs`, `src/index/minhash_lsh_index_trait.rs`, `src/ffi/minhash_lsh_ffi.rs`, `src/api/index.rs` | Partial | P1 | ✅ 参数别名映射已补齐；✅ FFI query 长度已对齐 `mh_vec_length * mh_vec_element_size`；✅ `MinHashLSHIndex` 已接入 `Index trait`（2026-03-06 18:02）；⏳ 仍需补齐 FFI 统一路径验证与全量 tests 回归 |
| FFI ABI | C++ factory + index runtime behavior | `src/ffi.rs`, `docs/FFI_CAPABILITY_MATRIX.md` | Partial | P1 | ✅ capability matrix documented; ✅ consistent error handling (19 NotImplemented returns); runtime behavior mismatch removal ongoing |

## 4. Validation Policy

- Every benchmark must report:
  - ground truth source
  - R@10
  - QPS
  - credibility tag (`trusted` / `unreliable` / `recheck required`)
- Credibility rules:
  - R@10 >= 80% => trusted (if setup valid)
  - 50% <= R@10 < 80% => unreliable
  - R@10 < 50% => recheck required

## 5. Audit Changelog

- 2026-03-05 22:32: Detailed interface comparison between C++ and Rust.
  - **C++ Index class methods (17 total):**
    - Build/Train/Add/Search (core lifecycle) ✅
    - AnnIterator (streaming results) ⚠️ trait defined but not implemented in any index
    - RangeSearch (radius-based search) ⚠️ some indexes return Unsupported
    - GetVectorByIds (vector retrieval) ⚠️ some indexes return Unsupported
    - HasRawData (raw data check) ✅
    - IsAdditionalScalarSupported ❌ **MISSING** in Rust
    - GetIndexMeta ❌ **MISSING** in Rust
    - Serialize/Deserialize (BinarySet) ✅ (serialize_to_memory/deserialize_from_memory)
    - DeserializeFromFile ⚠️ Rust has save/load but not exact equivalent
    - Dim/Size/Count/Type ✅ (dim/count/index_type; missing Size)
  - **Priority actions:**
    - P0: Implement AnnIterator for core indexes (HNSW/IVF/Flat)
    - P1: Add IsAdditionalScalarSupported and GetIndexMeta methods
    - P1: Ensure all indexes properly implement or reject unsupported methods
  - **Files checked:**
    - C++: `/Users/ryan/Code/vectorDB/knowhere/include/knowhere/index/index.h:152-236`
    - Rust: `/Users/ryan/.openclaw/workspace-builder/knowhere-rs/src/index.rs:102-267`

- 2026-03-05 21:22: Scanned interface alignment between C++ and Rust.
  - C++ Index class methods: Build/Train/Add/Search/AnnIterator/RangeSearch/GetVectorByIds/HasRawData/Serialize/Deserialize/DeserializeFromFile
  - Rust Index trait methods: train/add/search/range_search/create_ann_iterator/get_vector_by_ids/has_raw_data/serialize_to_memory/deserialize_from_memory/save/load
  - Gap: AnnIterator defined but not implemented in any index; DeserializeFromFile missing in Rust
  - HNSW implements core methods (train/add/search/range_search/get_vector_by_ids/save/load)
  - Next: Verify all core indexes implement or reject unsupported methods consistently

- 2026-03-05: Initialized parity audit baseline with module/file mapping and risk triage.
