# PQFlash DiskANN Sectioned Snapshot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development when subagent capacity is available, otherwise execute directly with the same checkpoint discipline.

**Goal:** Add a sectioned snapshot path for `PQFlashIndex` so DiskANN graph state can be stored as inspectable sections instead of only as the current file-group layout.

**Architecture:** Implement an owned-memory v1 snapshot for `PQFlashIndex`. The format carries Vamana graph arrays, raw vectors, external IDs, entry points, config/layout metadata, and optional PQ encoder/code bytes. The loader materializes an in-memory `PQFlashIndex` with no `DiskStorage`/mmap state. This keeps the first DiskANN ABI path runnable while leaving mmap/page-cache/lazy modes for a later section-backed runtime.

---

## Scope

In scope:

- `PQFlashIndex` sectioned export/write/load for owned-memory runtime.
- Vamana graph arrays:
  - `node_ids`
  - `node_neighbor_counts`
  - `node_neighbor_ids`
  - `entry_points`
  - `flat_stride`
- raw vector mirror for exact/rerank path.
- optional ProductQuantizer metadata and PQ codes.
- deleted row IDs.
- registry integration for `pqflash_sections_v1`.

Out of scope:

- `LoadMode::Mmap`, `LoadMode::PageCache`, and `LoadMode::Lazy` section-backed runtimes.
- HVQ and SQ8 prefilter state. v1 rejects these states on export instead of producing incomplete snapshots.
- `DiskAnnSqIndex` and `DiskAnnPcaUsqIndex` wrapper formats.

---

## Section Layout v1

Variant:

```text
pqflash_sections_v1
```

Sections:

```text
pqflash.meta.json
pqflash.vectors.f32
pqflash.node_ids.i64
pqflash.neighbor_counts.u32
pqflash.neighbor_ids.u32
pqflash.node_pq_codes.u8
pqflash.deleted_rows.u64
pqflash.pq_centroids.f32
```

Metadata fields:

```text
version
dim
metric
count
trained
flat_stride
pq_code_size
config
flash_layout
entry_points
pq_m
pq_nbits
```

Rules:

- `vectors.len == count * dim`.
- `node_ids.len == count`.
- `neighbor_counts.len == count`.
- `neighbor_ids.len == count * flat_stride`.
- each neighbor count `<= flat_stride` and `<= config.max_degree`.
- each non-padding neighbor ID `< count`.
- `node_pq_codes.len == count * pq_code_size` when `pq_code_size > 0`; otherwise empty.
- `pq_centroids` must be present iff `pq_code_size > 0`.
- deleted rows must be `< count`.

---

## Tasks

- [x] Add `PQFlashSectionedExport` and `PQFlashIndex::export_sectioned_snapshot`.
- [x] Add `PQFlashIndex::from_sectioned_snapshot_export`.
- [x] Add `PqFlashSectionedSnapshot` writer/loader.
- [x] Add `PQFlashIndex` `AnnRuntime` implementation for registry use.
- [x] Register `pqflash_sections_v1` in `default_ann_snapshot_registry`.
- [x] Add tests for section shape, memory/file roundtrip, registry load, and malformed graph rejection.

---

## Verification

Run:

```bash
cargo test --test test_pqflash_sectioned_snapshot -- --nocapture
cargo test --test test_ann_snapshot_registry -- --nocapture
cargo test --lib faiss::diskann_aisaq::tests -- --nocapture
cargo fmt --all -- --check
```
