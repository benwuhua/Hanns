# FFI Capability Matrix

Last updated: 2026-03-22

## Purpose

Document the capability matrix for all FFI-exposed index types, showing which operations are supported.

> Note: `docs/PARITY_AUDIT.md` is the authoritative audit log. This matrix is a condensed operator-facing view and must stay consistent with the audit’s `supported / constrained / unsupported` wording.
> Note: production acceptance still uses the remote x86 authority machine. Local `cargo test` output is useful for prefiltering but does not by itself close a contract change.

## Capability Legend

- ✅ Fully implemented and tested
- ⚠️ Partially implemented or has known issues
- ❌ Not implemented
- 🔄 Planned/In progress

## Index Type Capabilities

| Index Type | Train | Add | Search | Range Search | Ann Iterator | Get By ID | File Save/Load | Memory Serialize | DeserializeFromFile |
|---|---|---|---|---|---|---|---|---|---|
| Flat | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | ✅ | ✅ | ✅ |
| HNSW | ✅ | ✅ | ✅ | ⚠️ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ScaNN | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ✅ | ❌ | ✅ |
| HNSW-PRQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ |
| IVF-RaBitQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| HNSW-SQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| HNSW-PQ | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| DiskANN | ✅ | ✅ | ✅ | ❌ | ✅ | ⚠️ | ❌ | ❌ | ❌ |
| IVF-Flat | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| IVF-SQ8 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ |
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
- DiskANN currently exposes `get_vector_by_ids` only under its raw-data metric gate (`L2` / `Cosine`). Its `PQCode` branch in `src/faiss/diskann.rs` is regression-locked as an explicit mean-quantization placeholder rather than native-comparable PQ compression, so capability support here should be read as API semantics, not proof of production-grade DiskANN parity.
- HNSW FFI `get_vector_by_ids` is now wired through the same raw-vector path as the Rust `HnswIndex` advanced-path contract; metadata, direct FFI retrieval, and file save/load all agree on `supported`.

### Serialization
- `PERSIST-P3-003` 已把 `file_save_load` / `memory_serialize` / `deserialize_from_file` 的 supported / constrained / unsupported 语义重新拉齐到 audit 基线。
- 当前矩阵里 `File Save/Load` 与 `DeserializeFromFile` 表示“FFI/运行时已有稳定 contract”，不要求所有索引都支持内存序列化。
- 2026-03-22 更新：`HNSW` memory serialize 已接通；`IVF-SQ8` 已接通 file/memory/deserialize_from_file；`IVF-Flat`（float）已作为独立 C FFI 类型接入并支持 file/memory/deserialize_from_file。
- `HNSW-PQ` 继续维持稳定 `Unsupported`：`has_raw_data=false`，`get_vector_by_ids` 与 persistence 都是显式受限语义，而不是待补实现。
- `SparseWand` 现已具备文件级 save/load 与 `DeserializeFromFile` contract；`SparseWandCC` 仍不纳入统一 persistence 承诺。

## JNI Layer

当前 JNI `RegisteredIndex` 已接入以下索引：
- `Flat` (`MemIndex`)
- `HNSW`
- `IVF-Flat`
- `IVF-PQ`
- `IVF-SQ8`
- `DiskANN`

其中 `IVF-Flat` 与 `IVF-SQ8` 已接入 JNI 序列化/反序列化流程（内存字节路径）。

### Observability / Trace / Resource Contract
- `OBS-P3-005` 已把最小 runtime governance contract 收口到 `knowhere_get_index_meta` 返回 JSON。
- 新增三个稳定 section：
  - `observability`: 声明统一的 `knowhere.index.build` / `knowhere.index.search` / `knowhere.index.load` 事件名，以及 `latency_ms`、`topk`、`query_count`、`ground_truth_source`、`recall_at_10`、`artifact_path`、`mmap_load` 等字段口径。
  - `trace_propagation`: 固定 FFI 透传入口 `index_meta.trace_context_json`、gate runner 环境变量 `OPENCLAW_TRACE_CONTEXT_JSON`、以及 `w3c-traceparent-json` 编码约定。
  - `resource_contract`: 固定 `memory_bytes` / `disk_bytes` / `mmap_supported` / `unsupported_reason` 四元组，作为后续远端 build/test/perf gate 的最小资源审计基线。
- 本轮定义的是 contract，不等于已经接通完整 OpenTelemetry/Prometheus；真正的远端 tracing integration 与 production metrics 仍应在后续专项轮次落地。

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

Use local commands as quick smoke only, then replay the relevant gate on the remote x86 authority machine.

Local smoke:
```bash
cargo test --lib ffi -- --nocapture
cargo test serialize --lib
```

Authority replay:
```bash
bash init.sh
bash scripts/remote/test.sh --command "cargo test --lib ffi -- --nocapture"
bash scripts/remote/test.sh --command "cargo test --lib serialize -- --nocapture"
```

When a feature-specific contract also depends on artifact export or metadata shape, replay the matching regression from `feature-list.json` as well, for example:

```bash
bash scripts/remote/test.sh --command "cargo test --test bench_json_export -q"
```

This matrix is a contract view, not a performance-leadership claim. Family-level verdicts still come from the remote benchmark artifacts under `benchmark_results/`.

## Changes

- 2026-03-22: Updated matrix for latest FFI/JNI integration:
  - HNSW `memory_serialize` -> ✅
  - IVF-SQ8 `file_save_load`/`memory_serialize`/`deserialize_from_file` -> ✅
  - Added IVF-Flat (float) C FFI row with full persistence support
  - Added JNI Layer summary for Flat/HNSW/IVF-Flat/IVF-PQ/IVF-SQ8/DiskANN
- 2026-03-06 01:35: Added FFI AnnIterator interface (`knowhere_create_ann_iterator`/`knowhere_ann_iterator_next`/`knowhere_free_ann_iterator`), supports HNSW/ScaNN/HNSW-PQ
- 2026-03-06: Updated AnnIterator status for HNSW/ScaNN/HNSW-PQ/DiskANN (now ✅); HNSW GetByID ✅; ScaNN GetByID ⚠️
- 2026-03-08: Marked HNSW-PQ advanced-path semantics as constrained and stable: AnnIterator ✅, `get_vector_by_ids` ⚠️ (stable Unsupported due to lossy PQ), save/load ⚠️ (stable Unsupported pending persistence)
- 2026-03-05: Initial matrix creation, added AnnIterator interface
