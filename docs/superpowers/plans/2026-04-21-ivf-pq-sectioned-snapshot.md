# IVF-PQ Sectioned Snapshot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development when subagent capacity is available, otherwise execute directly with the same checkpoint discipline.

**Goal:** Add a sectioned snapshot path for `IvfPqIndex` so the Hanns storage ABI covers product-quantized IVF runtimes and exposes ANN-kernel state without relying on legacy monolithic byte streams.

**Architecture:** Mirror the IVF-Flat and IVF-SQ8 section model, but make PQ codebook state first-class. A runnable IVF-PQ snapshot must carry coarse centroids, PQ centroids, list offsets/sizes, list IDs, list PQ codes, raw vector mirror, and whether the PQ centroid table came from FAISS import. The loader materializes today’s `IvfPqIndex`, sets the PQ centroids directly, preserves imported centroid routing, and rebuilds compact inverted lists.

**Tech Stack:** Rust 2021, `src/faiss/ivfpq.rs`, `src/faiss/ivfpq_snapshot.rs`, existing `src/storage/*`, `serde_json`, memory/file artifact stores.

---

## Scope

In scope:

- `IvfPqIndex` sectioned export/write/load.
- PQ centroid preservation, including `imported_pq_centroids` state.
- Memory and file artifact roundtrips.
- Corrupt section validation.
- Existing IVF-PQ legacy byte/file persistence compatibility.

Out of scope:

- OPQ restore support. Current `IvfPqIndex::new` disables OPQ and `OptimizedProductQuantizer` has no direct restore API for rotation/inverse/codebook state.
- Lance/HannsDB/pgvector adapter changes.
- Changing legacy IVF-PQ save/load semantics.

---

## Section Layout v1

Variant:

```text
ivf_pq_sections_v1
```

Sections:

```text
ivf_pq.meta.json
ivf_pq.centroids.f32
ivf_pq.pq_centroids.f32
ivf_pq.list_offsets.u64
ivf_pq.list_sizes.u64
ivf_pq.list_ids.i64
ivf_pq.list_codes.u8
ivf_pq.ids.i64
ivf_pq.vectors.f32
```

Metadata fields:

```text
version
dim
metric
nlist
nprobe
m
nbits_per_idx
code_size
count
next_id
trained
use_opq
imported_pq_centroids
```

Rules:

- `trained == true` for runnable snapshots.
- `use_opq == false` in v1; reject OPQ snapshots until OPQ restore state is sectioned.
- `centroids.len == nlist * dim`.
- `pq_centroids.len == m * (1 << nbits_per_idx) * (dim / m)`.
- `code_size == (m * nbits_per_idx).div_ceil(8)`.
- `ids.len == count`.
- `vectors.len == count * dim`; imported FAISS snapshots may have `count == 0` raw vectors only if existing runtime does.
- `list_offsets.len == nlist`.
- `list_sizes.len == nlist`.
- `sum(list_sizes) == list_ids.len`.
- `list_codes.len == list_ids.len * code_size`.
- list offsets must be canonical contiguous offsets.

---

## Tasks

### Task 1: Export Surface

Files:

- Modify: `src/faiss/ivfpq.rs`
- Test: `tests/test_ivfpq_sectioned_snapshot.rs`

Steps:

- [x] Add `IvfPqSectionedExport`.
- [x] Add `IvfPqIndex::export_sectioned_snapshot() -> Result<IvfPqSectionedExport>`.
- [x] Add export shape test, including PQ centroids and code size.

### Task 2: Sectioned Writer

Files:

- Create: `src/faiss/ivfpq_snapshot.rs`
- Modify: `src/faiss/mod.rs`
- Test: `tests/test_ivfpq_sectioned_snapshot.rs`

Steps:

- [x] Add section constants and `IVF_PQ_SECTIONS_SNAPSHOT_VARIANT`.
- [x] Add `IvfPqSectionedSnapshot::from_index`.
- [x] Implement `AnnSnapshot` writer.
- [x] Re-export symbols from `src/faiss/mod.rs`.
- [x] Add expected-section and manifest-length tests.

### Task 3: Loader

Files:

- Modify: `src/faiss/ivfpq.rs`
- Modify: `src/faiss/ivfpq_snapshot.rs`
- Test: `tests/test_ivfpq_sectioned_snapshot.rs`

Steps:

- [x] Add `IvfPqIndex::from_sectioned_snapshot_export`.
- [x] Validate all layout rules and reject malformed artifacts.
- [x] Add `load_ivf_pq_index_from_artifact`.
- [x] Add memory-store search roundtrip test.
- [x] Add bitset roundtrip test with explicit IDs.

### Task 4: File Helpers and Compatibility

Files:

- Modify: `src/faiss/ivfpq_snapshot.rs`
- Test: `tests/test_ivfpq_sectioned_snapshot.rs`

Steps:

- [x] Add `save_ivf_pq_sectioned_snapshot`.
- [x] Add `load_ivf_pq_sectioned_snapshot`.
- [x] Add file-store roundtrip test.
- [x] Add corrupt-section tests for bad PQ centroid length, bad list code length, and descriptor length mismatch.
- [x] Confirm existing IVF-PQ legacy save/load tests still pass.

---

## Verification

Run:

```bash
cargo test --test test_ivfpq_sectioned_snapshot -- --nocapture
cargo test --lib faiss::ivfpq::tests -- --nocapture
cargo test --test test_ivf_sq8_sectioned_snapshot -- --nocapture
cargo fmt --all -- --check
```

Expected:

- Sectioned IVF-PQ snapshots restore runnable search-equivalent runtimes.
- PQ centroid state is restored without retraining.
- Existing IVF-PQ legacy persistence remains unchanged.
- No Lance/HannsDB/pgvector files are modified.
