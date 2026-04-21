# DiskANN-SQ Sectioned Snapshot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development when subagent capacity is available, otherwise execute directly with the same checkpoint discipline.

**Goal:** Add a sectioned snapshot path for `DiskAnnSqIndex` so wrapper-level SQ/PCA state can be stored alongside the sectioned PQFlash graph ABI.

**Architecture:** Reuse `PQFlashSectionedExport` for the inner DiskANN graph and add wrapper sections for SQ8 quantizer, SQ codes, optional PCA transform, and wrapper config. The loader reconstructs `PQFlashIndex` from the embedded export and then materializes `DiskAnnSqIndex`.

---

## Scope

In scope:

- `DiskAnnSqIndex` sectioned export/write/load.
- SQ8 quantizer and code payload preservation.
- optional PCA transform preservation.
- memory/file artifact roundtrips.
- graph corruption and SQ code length validation.

Out of scope:

- `DiskAnnPcaUsqIndex`.
- section-backed PQFlash mmap/page-cache/lazy runtimes.
- changing existing `DiskAnnSqIndex::save/load` file format.

---

## Section Layout v1

Variant:

```text
diskann_sq_sections_v1
```

Sections:

```text
diskann_sq.meta.json
diskann_sq.sq.meta.json
diskann_sq.sq.codes.u8
diskann_sq.pca.meta.json
diskann_sq.pca.mean.f32
diskann_sq.pca.components.f32
```

The sectioned snapshot embeds the inner `PqFlashSectionedSnapshot` by prefixing its manifest and sections with `diskann_sq.inner.*`, so large graph/vector/PQ arrays remain section-addressable rather than being collapsed into wrapper metadata.

---

## Tasks

- [x] Map wrapper fields and inner PQFlash reuse path.
- [x] Add `DiskAnnSqSectionedExport` and `DiskAnnSqIndex::export_sectioned_snapshot`.
- [x] Add `DiskAnnSqIndex::from_sectioned_snapshot_export`.
- [x] Add `DiskAnnSqSectionedSnapshot` writer/loader/file helpers.
- [x] Add tests for memory/file roundtrip, PCA roundtrip, and bad SQ code length.

---

## Verification

Run:

```bash
cargo test --test test_diskann_sq_sectioned_snapshot -- --nocapture
cargo test --lib faiss::diskann_sq::tests -- --nocapture
cargo test --test test_pqflash_sectioned_snapshot -- --nocapture
cargo fmt --all -- --check
```
