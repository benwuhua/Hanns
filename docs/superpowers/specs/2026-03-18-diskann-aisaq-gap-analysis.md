# DiskANN/PQFlash Gap Analysis & Long-Term Task List

**Date**: 2026-03-18
**Scope**: knowhere-rs `diskann_aisaq.rs` vs native knowhere C++ + Rust DiskANN reference
**Reference codebases**:
- Native C++: `/Users/ryan/Code/knowhere/src/index/diskann/diskann.cc`
- Rust DiskANN: `/Users/ryan/Code/DiskANN/` (Microsoft Rust port)

---

## Context: Current Status

### Session work completed (2026-03-18)
| Item | Before | After |
|------|--------|-------|
| `materialize_storage()` disk QPS | 648 QPS | 30,685 QPS (+47x, Mac) |
| Parallel PQ k-means | 114.9s | 25.4s (+4.5x) |
| filter_threshold + bitset lanes | ❌ | ✅ |
| rearrange/rerank switch | ❌ | ✅ |
| warm-up / PQ-cache config | ❌ | ✅ |
| pq_code_budget enforcement | ❌ | ✅ |
| 1M benchmark + x86 remote | ❌ | ✅ |

### 1M Benchmark Results (2026-03-18)

| Platform | Index | Build | QPS |
|----------|-------|-------|-----|
| Mac (M-series) | NoPQ 1M | 67.9s | 21,921 |
| Mac (M-series) | PQ32 1M | 99.0s | 18,431 |
| x86 (8c) | NoPQ 1M | 116.8s | 9,722 |
| x86 (8c) | PQ32 1M | 238.6s | 7,673 |

### Formal project verdicts (from prior Codex sessions)
- HNSW: ✅ **leading** (Rust 1.789x faster than native at near-equal recall)
- IVF-PQ: ❌ **no-go** (recall < 0.8 gate)
- DiskANN: ⚠️ **constrained** (functional but not native-comparable)

---

## Gap Analysis: AISAQ vs Reference Implementations

### 1. Search capability gaps (vs native knowhere)

| Feature | Native knowhere | AISAQ current | Gap? |
|---------|----------------|---------------|------|
| Beam search (L + beamwidth) | ✅ | ✅ | — |
| BitsetView soft-delete filter | ✅ | ✅ | — |
| filter_threshold (PQ+Refine) | ✅ | ✅ | — |
| Warm-up | ✅ | ✅ | — |
| PQ code budget | ✅ | ✅ | — |
| **Post-beam exact rerank** | ✅ (inside `cached_beam_search`) | partial (`rearrange_candidates` switch exists but uses ADC, not exact) | **P0** |
| **RangeSearch** | ✅ (with `range_filter`, `search_list_and_k_ratio`) | ❌ | **P1** |
| `for_tuning` flag | ✅ | ❌ | P2 |

### 2. Data / ID management gaps (vs both references)

| Feature | Native | Rust DiskANN | AISAQ | Gap? |
|---------|--------|-------------|-------|------|
| External ID passthrough | DataSet.SetIds/GetIds | `DataProvider` trait (to_internal/external_id) | ❌ (row-id only) | **P1** |
| Soft delete via BitsetView | ✅ (logical only) | ✅ (`ElementStatus::Deleted`) | partial (BitsetView passed through, not stored) | **P1** |
| **Incremental insert** | ❌ (immutable) | ✅ (`add()` on live index) | ❌ | **P2** |
| **Hard delete + consolidation** | ❌ (logical only) | ✅ (`consolidate_vector()`) | ❌ | **P2** |
| GetVectorByIds (raw data) | ✅ (if NoPQ) | — | partial | P2 |

### 3. Storage / persistence gaps

| Feature | Native | AISAQ | Gap? |
|---------|--------|-------|------|
| **On-disk persistence** (FileManager) | ✅ (file-based serialize) | ❌ (in-memory only, lost on restart) | **P1** |
| Disk-backed graph (cold path) | ✅ (page cache) | ✅ (page cache, or materialize to RAM) | — |
| Aligned async IO (io_uring) | ✅ (Linux AIO) | ❌ (synchronous reads) | P2 |
| PQ pivot / compressed vector files | ✅ | ✅ (in-memory PQ) | P1 (disk layout) |

### 4. Performance gaps

| Metric | x86 (8c) | Mac M-series | Root cause |
|--------|----------|--------------|------------|
| NoPQ 1M QPS | 9,722 | 21,921 | Memory bandwidth (Apple UMA vs DDR5) |
| PQ k-means train 1M | 127.6s | 49.6s | Same: memory bandwidth |
| Disk cached QPS | 13,852 | 30,685 | Same |
| AVX-512 utilization | unknown | N/A | Need to verify SIMD detection on x86 |

---

## Long-Term Task List (Prioritized)

### P0 — Immediate (unblocks recall parity with native)

#### AISAQ-CAP-001: True exact rerank stage
- **What**: After PQ beam search, fetch raw vectors for top-N candidates and re-sort by exact distance
- **Why**: Native `cached_beam_search` always does this internally. Without it, PQ quantization error bleeds into final results → recall ceiling
- **Reference**: `diskann-disk/src/search/provider/disk_provider.rs` `post_process()` — fetches raw floats, recomputes L2, sorts
- **Scope**: `diskann_aisaq.rs` `search_internal()` — after `beam_search_io`, take top rerank_k candidates, compute exact distances, re-sort
- **Exit criterion**: Recall@10 improves measurably on 10K/100K benchmark; authority A/B confirms

### P1 — Before Milvus integration

#### AISAQ-CAP-002: External ID / Tag system
- **What**: `int64` external_id → internal `row_id` bidirectional map, stored alongside the index
- **Why**: Milvus passes external IDs; search results must return them
- **Reference**: Native: DataSet.SetIds() passthrough. Rust DiskANN: `DataProvider` trait
- **Design**: `BTreeMap<i64, u32>` (external→internal) + `Vec<i64>` (internal→external); persist alongside graph
- **Scope**: New `IdMap` struct; wire into `add()`, `search()` result IDs, serialize/deserialize

#### AISAQ-CAP-003: On-disk persistence
- **What**: Serialize the full index (graph CSR, raw vectors or PQ codes, medoid list, config) to disk; reload without rebuild
- **Why**: Currently `materialize_storage()` builds in-memory flat arrays but they are lost on restart
- **Reference**: Native: FileManager-based file layout. Rust DiskANN: `DiskIndexWriter`
- **Design**: Write CSR arrays + PQ codebooks + raw vectors to memory-mapped files; load via `mmap`
- **Scope**: New `save(path)` / `load(path)` in `PQFlashIndex`; update `Index` trait impl

#### AISAQ-CAP-004: RangeSearch
- **What**: Return all vectors within distance `radius` from query
- **Reference**: Native: `DiskANNIndexNode::RangeSearch` with `range_filter`, `search_list_and_k_ratio`
- **Scope**: New `range_search()` method; reuse beam search with expanded L, filter results by distance threshold

### P2 — Production completeness

#### AISAQ-CAP-005: Incremental insert + lazy delete + consolidation
- **What**: Add vectors to a live index without full rebuild; mark deletes; periodically consolidate to reclaim slots
- **Reference**: Rust DiskANN `index.rs` `add()` / `consolidate_vector()` / `ElementStatus::Deleted`
- **Scope**: Major refactor of graph build path; needs concurrent-safe insert

#### AISAQ-CAP-006: Multi-entry-point medoid seeding
- **What**: Use multiple medoid candidates as search entry points instead of single medoid
- **Why**: Better recall on clustered data; already mentioned in `diskann_capability_closure_plan.md`
- **Scope**: `beam_search_io` entry-point selection; medoid list (already partially present)

#### AISAQ-CAP-007: x86 SIMD audit
- **What**: Verify AVX2/AVX-512 paths are activated on the remote x86 authority machine
- **Why**: x86 is 2.25x slower than Mac; part of gap may be missing SIMD (not just memory bandwidth)
- **Scope**: Check `simd.rs` CPUID dispatch; benchmark with/without explicit SIMD features; compare PQ ADC throughput

#### AISAQ-CAP-008: Async IO (io_uring) for cold-disk path
- **What**: Replace synchronous page-cache reads with async io_uring batched reads
- **Why**: Current cold-disk QPS ~330; async batched IO could reach 5-10K QPS
- **Scope**: New `AsyncBeamSearchIO` implementing `BeamSearchIO` with tokio + io_uring backend

### P3 — IVF-PQ recall fix (separate track)

#### IVFPQ-FIX-001: Residual PQ recall root cause
- **What**: IVF-PQ recall is < 0.8 (no-go) on all benchmark lanes
- **Hypothesis**: Residual computation bug or codebook assignment mismatch
- **Scope**: `ivfpq.rs` focused regression; compare with faiss reference encoding/decoding

---

## Completion Criteria per Task

Each task requires before it can be marked done:
1. Focused unit tests (red → green)
2. Authority benchmark A/B (remote x86): recall and QPS before/after
3. Update to `TASK_QUEUE.md` with verdict

---

## Notes on Reference Code

- **Native knowhere DiskANN is immutable** (no incremental insert): deletions are logical-only via BitsetView. Rust DiskANN is more capable here.
- **External IDs in native knowhere** are a DataSet-level concept, not stored inside the index. The mapping is maintained by Milvus above knowhere.
- **Rust DiskANN reference** uses async tokio for disk IO — this is the architecture to target for async IO (AISAQ-CAP-008).
- **AISAQ's `rearrange_candidates`** (added this session) is the right hook for CAP-001, but currently uses ADC not exact distance. The fix is to use raw float vectors for reranking.
