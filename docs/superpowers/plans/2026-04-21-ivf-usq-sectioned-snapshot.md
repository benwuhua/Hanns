# IVF-USQ Sectioned Snapshot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development when subagent capacity is available, otherwise execute directly with the same checkpoint discipline.

**Goal:** Add a sectioned snapshot path for `IvfUsqIndex` so the Hanns storage ABI covers packed/sign-bit quantized IVF runtimes without depending on the legacy bincode blob.

**Architecture:** Expose the USQ kernel state directly: IVF centroids, quantizer centroid/config, per-list IDs, per-list packed B-bit codes, per-list sign-bit fastscan codes, and per-vector metadata (`norm`, `norm_sq`, `vmax`, `quant_quality`). The loader rebuilds `pending` and `UsqLayout` from these sections, preserving query scoring state without retraining.

**Tech Stack:** Rust 2021, `src/faiss/ivf_usq.rs`, `src/faiss/ivf_usq_snapshot.rs`, existing `src/storage/*`, `serde_json`, memory/file artifact stores.

---

## Scope

In scope:

- `IvfUsqIndex` sectioned export/write/load.
- Quantizer centroid preservation.
- Packed/sign-bit payload preservation.
- Memory and file artifact roundtrips.
- Corrupt section validation.
- Existing IVF-USQ legacy file persistence compatibility.

Out of scope:

- Changing legacy `IvfUsqIndex::save/load` bincode format.
- Lance/HannsDB/pgvector adapter changes.
- Adding raw-vector reconstruction for lossy USQ indexes.

---

## Section Layout v1

Variant:

```text
ivf_usq_sections_v1
```

Sections:

```text
ivf_usq.meta.json
ivf_usq.centroids.f32
ivf_usq.quantizer_centroid.f32
ivf_usq.list_offsets.u64
ivf_usq.list_sizes.u64
ivf_usq.list_ids.i64
ivf_usq.packed_bits.u8
ivf_usq.sign_bits.u8
ivf_usq.norms.f32
ivf_usq.norms_sq.f32
ivf_usq.vmaxs.f32
ivf_usq.quant_qualities.f32
```

Metadata fields:

```text
version
dim
metric
nlist
nprobe
bits_per_dim
rotation_seed
rerank_k
use_high_accuracy_scan
ntotal
trained
code_bytes
sign_bytes
padded_dim
```

Rules:

- `trained == true` for runnable snapshots.
- `centroids.len == nlist * dim`.
- `quantizer_centroid.len == dim`.
- `list_offsets.len == nlist`.
- `list_sizes.len == nlist`.
- `sum(list_sizes) == ntotal == list_ids.len`.
- `packed_bits.len == ntotal * code_bytes`.
- `sign_bits.len == ntotal * sign_bytes`.
- `norms.len == norms_sq.len == vmaxs.len == quant_qualities.len == ntotal`.
- list offsets must be canonical contiguous offsets.

---

## Tasks

### Task 1: Export Surface

Files:

- Modify: `src/faiss/ivf_usq.rs`
- Test: `tests/test_ivf_usq_sectioned_snapshot.rs`

Steps:

- [x] Add `IvfUsqSectionedExport`.
- [x] Add `IvfUsqIndex::export_sectioned_snapshot() -> Result<IvfUsqSectionedExport>`.
- [x] Add export shape test including code/sign-byte sizes and quantizer centroid.

### Task 2: Sectioned Writer

Files:

- Create: `src/faiss/ivf_usq_snapshot.rs`
- Modify: `src/faiss/mod.rs`
- Test: `tests/test_ivf_usq_sectioned_snapshot.rs`

Steps:

- [x] Add section constants and `IVF_USQ_SECTIONS_SNAPSHOT_VARIANT`.
- [x] Add `IvfUsqSectionedSnapshot::from_index`.
- [x] Implement `AnnSnapshot` writer.
- [x] Re-export symbols from `src/faiss/mod.rs`.
- [x] Add expected-section and manifest-length tests.

### Task 3: Loader

Files:

- Modify: `src/faiss/ivf_usq.rs`
- Modify: `src/faiss/ivf_usq_snapshot.rs`
- Test: `tests/test_ivf_usq_sectioned_snapshot.rs`

Steps:

- [x] Add `IvfUsqIndex::from_sectioned_snapshot_export`.
- [x] Validate all layout rules and reject malformed artifacts.
- [x] Add `load_ivf_usq_index_from_artifact`.
- [x] Add memory-store search roundtrip test.

### Task 4: File Helpers and Compatibility

Files:

- Modify: `src/faiss/ivf_usq_snapshot.rs`
- Test: `tests/test_ivf_usq_sectioned_snapshot.rs`

Steps:

- [x] Add `save_ivf_usq_sectioned_snapshot`.
- [x] Add `load_ivf_usq_sectioned_snapshot`.
- [x] Add file-store roundtrip test.
- [x] Add corrupt-section tests for bad centroid length, bad packed length, and descriptor length mismatch.
- [x] Confirm existing IVF-USQ tests still pass.

---

## Verification

Run:

```bash
cargo test --test test_ivf_usq_sectioned_snapshot -- --nocapture
cargo test --lib faiss::ivf_usq::tests -- --nocapture
cargo test --test test_ivfpq_sectioned_snapshot -- --nocapture
cargo fmt --all -- --check
```

Expected:

- Sectioned IVF-USQ snapshots restore runnable search-equivalent runtimes.
- Quantizer centroid is preserved explicitly.
- Existing IVF-USQ persistence remains unchanged.
- No Lance/HannsDB/pgvector files are modified.
