# PARITY_AUDIT (Non-GPU)

Last updated: 2026-03-05 23:35
Sync baseline: b284d55 from origin/main

## 轮次记录
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
| Core contract | `include/knowhere/index/index.h`, `include/knowhere/index/index_node.h` | `src/index.rs`, `src/api/search.rs` | Partial | P0 | unify lifecycle contract and fallback semantics; AnnIterator trait implemented (2026-03-05) |
| Index factory/legality | `include/knowhere/index/index_factory.h`, `include/knowhere/index/index_table.h`, `include/knowhere/comp/knowhere_check.h` | `src/api/index.rs`, `src/faiss/mod.rs`, `src/ffi.rs` | Partial | P1 | centralized legal matrix for index/datatype/metric |
| HNSW | `src/index/hnsw/faiss_hnsw.cc` | `src/faiss/hnsw.rs` | Partial | P1 | ✅ get_vector_by_ids (2026-03-05), ✅ AnnIterator (2026-03-05); range search parity remaining |
| IVF core | `src/index/ivf/ivf.cc`, `src/index/ivf/ivf_config.h` | `src/faiss/ivf.rs`, `src/faiss/ivf_flat.rs`, `src/faiss/ivfpq.rs`, `src/api/index.rs` | Partial | P1 | parameter coverage and edge behavior alignment; SIMD slice fix in ivf_sq_cc (2026-03-05) |
| RaBitQ | `src/index/ivf/ivfrbq_wrapper.*` | `src/faiss/ivf_rabitq.rs`, `src/faiss/rabitq_ffi.rs` | Partial | P1 | query-bits and config boundary consistency |
| DiskANN | `src/index/diskann/diskann.cc`, `src/index/diskann/diskann_config.h` | `src/faiss/diskann.rs`, `src/faiss/diskann_complete.rs` | Partial | P1 | ✅ AnnIterator (inherent impl); lifecycle parity and config semantics; add_batch SIMD slice fix (2026-03-05) |
| AISAQ | `src/index/diskann/diskann_aisaq.cc`, `src/index/diskann/aisaq_config.h` | `src/faiss/diskann_aisaq.rs`, `src/faiss/aisaq.rs` | Partial | P1 | parameter and file-layout behavior alignment |
| ScaNN | - | `src/faiss/scann.rs` | Partial | P1 | ✅ AnnIterator (2026-03-05), get_vector_by_ids, has_raw_data |
| HNSW-PQ | - | `src/faiss/hnsw_pq.rs` | Partial | P2 | ✅ AnnIterator (2026-03-05); has_raw_data=false (lossy) |
| Sparse | `src/index/sparse/sparse_index_node.cc`, `src/index/sparse/sparse_inverted_index.h` | `src/faiss/sparse_inverted.rs`, `src/faiss/sparse_wand.rs`, `src/faiss/sparse_wand_cc.rs` | Partial | P2 | iterator/filter behavior and parameter parity |
| MinHash | `src/index/minhash/minhash_index_node.cc`, `src/index/minhash/minhash_lsh_config.h` | `src/index/minhash_lsh.rs`, `src/ffi/minhash_lsh_ffi.rs`, `src/api/index.rs` | Partial | P2 | `mh_*` parameter naming/validation parity |
| FFI ABI | C++ factory + index runtime behavior | `src/ffi.rs`, `docs/FFI_CAPABILITY_MATRIX.md` | Partial | P0 | capability matrix documented; runtime behavior mismatch removal ongoing |

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
