# FFI Capability Matrix

Last updated: 2026-03-05 20:40

## Purpose

Document the capability matrix for all FFI-exposed index types, showing which operations are supported.

## Capability Legend

- ✅ Fully implemented and tested
- ⚠️ Partially implemented or has known issues
- ❌ Not implemented
- 🔄 Planned/In progress

## Index Type Capabilities

| Index Type | Train | Add | Search | Range Search | Ann Iterator | Get By ID | Serialize | Deserialize |
|---|---|---|---|---|---|---|---|---|
| Flat | ✅ | ✅ | ✅ | ⚠️ | ❌ | ✅ | ✅ | ✅ |
| HNSW | ✅ | ✅ | ✅ | ⚠️ | ❌ | ⚠️ | ✅ | ✅ |
| ScaNN | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| HNSW-PRQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| IVF-RaBitQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| HNSW-SQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| HNSW-PQ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| IVF-SQ8 | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| BinFlat | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | ✅ | ✅ |
| BinaryHNSW | ✅ | ✅ | ✅ | ⚠️ | ❌ | ❌ | ⚠️ | ⚠️ |
| BinIVF-Flat | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| SparseWand | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| SparseWandCC | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| MinHashLSH | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| IVF-Flat-CC | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ⚠️ | ⚠️ |
| IVF-SQ-CC | ✅ | ✅ | ✅ | ❌ | ❌ | ⚠️ | ⚠️ | ⚠️ |

## Notes

### Range Search
- Most indexes have basic range search implementation but may have edge cases
- Need comprehensive testing for radius-based filtering

### Ann Iterator
- New interface added (2026-03-05) to match C++ knowhere
- No indexes implement it yet
- Planned for HNSW and IVF families first

### Get Vector By ID
- Only indexes that store raw data (Flat, IVF-Flat variants) can fully support
- Quantization indexes (SQ8, PQ, RaBitQ) cannot return original vectors

### Serialization
- Basic file-based serialization implemented
- BinarySet-based memory serialization needs more work for some index types

## Priority for Completion

### P0 (Critical)
1. AnnIterator interface implementation for core indexes
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

- 2026-03-05: Initial matrix creation, added AnnIterator interface
