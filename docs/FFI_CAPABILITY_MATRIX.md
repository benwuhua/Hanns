# FFI Capability Matrix

Last updated: 2026-03-09 12:30

## Purpose

Document the capability matrix for all FFI-exposed index types, showing which operations are supported.

> Note: `docs/PARITY_AUDIT.md` is the authoritative audit log. This matrix is a condensed operator-facing view and must stay consistent with the audit’s `supported / constrained / unsupported` wording.

## Capability Legend

- ✅ Fully implemented and tested
- ⚠️ Partially implemented or has known issues
- ❌ Not implemented
- 🔄 Planned/In progress

## Index Type Capabilities

| Index Type | Train | Add | Search | Range Search | Ann Iterator | Get By ID | File Save/Load | Memory Serialize | DeserializeFromFile |
|---|---|---|---|---|---|---|---|---|---|
| Flat | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | ✅ | ✅ | ✅ |
| HNSW | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ❌ | ✅ |
| ScaNN | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ✅ | ❌ | ✅ |
| HNSW-PRQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| IVF-RaBitQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| HNSW-SQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| HNSW-PQ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| DiskANN | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| IVF-SQ8 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ❌ | ❌ |
| BinFlat | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| BinaryHNSW | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | ❌ | ❌ | ❌ |
| BinIVF-Flat | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| SparseWand | ❌ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ✅ | ❌ | ✅ |
| SparseWandCC | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| MinHashLSH | ❌ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ✅ | ❌ | ✅ |

## Notes

### Range Search
- Most indexes have basic range search implementation but may have edge cases
- Need comprehensive testing for radius-based filtering

### Ann Iterator
- Interface added (2026-03-05) to match C++ knowhere
- **FFI interface added (2026-03-06 01:35):** `knowhere_create_ann_iterator`/`knowhere_ann_iterator_next`/`knowhere_free_ann_iterator`
- **Implemented indexes (2026-03-05 23:35):** HNSW, ScaNN, HNSW-PQ, DiskANN
- **FFI support (2026-03-06 01:35):** HNSW, ScaNN, HNSW-PQ (DiskANN pending)
- Implementation locations:
  - HNSW: `src/faiss/hnsw.rs:2470`, FFI: `src/ffi.rs:2222`
  - ScaNN: `src/faiss/scann.rs:1005`, FFI: `src/ffi.rs:2222`
  - HNSW-PQ: `src/faiss/hnsw_pq.rs:729`, FFI: `src/ffi.rs:2222`
  - DiskANN: `src/faiss/diskann.rs:961` (inherent impl)
- Planned for IVF family next

### Get Vector By ID
- Only indexes that store raw data (Flat, IVF-Flat variants) can fully support
- Quantization indexes (SQ8, PQ, RaBitQ) cannot return original vectors
- HNSW-PQ is intentionally constrained: `has_raw_data=false`, and `get_vector_by_ids` returns a stable `Unsupported` contract because PQ storage is lossy

### Serialization
- `PERSIST-P3-003` 已把 `file_save_load` / `memory_serialize` / `deserialize_from_file` 的 supported / constrained / unsupported 语义重新拉齐到 audit 基线。
- 当前矩阵里 `File Save/Load` 与 `DeserializeFromFile` 表示“FFI/运行时已有稳定 contract”，不要求所有索引都支持内存序列化。
- `HNSW-PQ` 继续维持稳定 `Unsupported`：`has_raw_data=false`，`get_vector_by_ids` 与 persistence 都是显式受限语义，而不是待补实现。
- `SparseWand` 现已具备文件级 save/load 与 `DeserializeFromFile` contract；`SparseWandCC` 仍不纳入统一 persistence 承诺。

## Priority for Completion

### P0 (Critical)
1. ✅ AnnIterator FFI interface implementation (DONE 2026-03-06)
2. Verify serialize/deserialize for all index types
3. Complete GetVectorByIds where applicable

### P1 (Important)
1. Range search edge case handling
2. Binary serialization (BinarySet) for all indexes
3. Error path consistency

### P2 (Enhancement)
1. Performance optimization for serialization
2. Compression for serialized data
3. Streaming iterators

## Validation

Run `cargo test` to verify all implemented capabilities work correctly.

For specific index type testing:
```bash
# Test specific index family
cargo test hnsw --lib
cargo test ivf --lib

# Test serialization
cargo test serialize --lib
```

## Changes

- 2026-03-06 01:35: Added FFI AnnIterator interface (`knowhere_create_ann_iterator`/`knowhere_ann_iterator_next`/`knowhere_free_ann_iterator`), supports HNSW/ScaNN/HNSW-PQ
- 2026-03-06: Updated AnnIterator status for HNSW/ScaNN/HNSW-PQ/DiskANN (now ✅); HNSW GetByID ✅; ScaNN GetByID ⚠️
- 2026-03-08: Marked HNSW-PQ advanced-path semantics as constrained and stable: AnnIterator ✅, `get_vector_by_ids` ⚠️ (stable Unsupported due to lossy PQ), save/load ⚠️ (stable Unsupported pending persistence)
- 2026-03-05: Initial matrix creation, added AnnIterator interface
