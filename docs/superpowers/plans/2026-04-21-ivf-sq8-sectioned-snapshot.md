# IVF-SQ8 Sectioned Snapshot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development when subagent capacity is available, otherwise execute this plan directly with the same test/review checkpoints.

**Goal:** Add a sectioned snapshot path for `IvfSq8Index` so the Hanns storage ABI covers a quantized IVF runtime, while preserving existing `IvfSq8Index::save/load/serialize_to_bytes/deserialize_from_bytes` behavior.

**Architecture:** Extend the IVF-Flat section vocabulary with quantized payload sections. The sectioned layout must carry the data needed by the ANN kernel directly: coarse centroids, list offsets/sizes, list IDs, list SQ8 residual codes, SQ8 quantizer parameters, raw vector mirror, and internal row mapping for bitset correctness.

**Tech Stack:** Rust 2021, `src/faiss/ivf_sq8.rs`, `src/faiss/ivf_sq8_snapshot.rs`, existing `src/storage/*`, existing `src/kernel/*`, `serde_json`, memory/file artifact stores.

---

## Scope

In scope:

- `IvfSq8Index` sectioned export/write/load.
- SQ8 quantizer parameter preservation.
- Inverted-list row mapping preservation for bitset filtering.
- Memory and file artifact roundtrips.
- Corrupt section validation.
- Existing IVF-SQ8 legacy byte/file persistence compatibility.

Out of scope:

- IVF-USQ and IVF-PQ section layouts.
- Lance/HannsDB/pgvector adapter changes.
- Changing legacy IVF-SQ8 save/load semantics.

---

## Section Layout v1

Variant:

```text
ivf_sq8_sections_v1
```

Sections:

```text
ivf_sq8.meta.json
ivf_sq8.centroids.f32
ivf_sq8.list_offsets.u64
ivf_sq8.list_sizes.u64
ivf_sq8.list_ids.i64
ivf_sq8.list_rows.u64
ivf_sq8.list_codes.u8
ivf_sq8.ids.i64
ivf_sq8.vectors.f32
```

Metadata fields:

```text
version
dim
metric
nlist
nprobe
count
next_id
trained
quantizer_bit
quantizer_min
quantizer_max
quantizer_scale
quantizer_offset
```

Rules:

- `trained == true` for runnable snapshots.
- `centroids.len == nlist * dim`.
- `ids.len == count`.
- `vectors.len == count * dim`.
- `list_offsets.len == nlist`.
- `list_sizes.len == nlist`.
- `sum(list_sizes) == list_ids.len == list_rows.len`.
- `list_codes.len == list_ids.len * dim`.
- each `list_offsets[i] + list_sizes[i] <= list_ids.len`.
- each `list_rows[j] < count`.
- each list row maps back to the same external ID: `ids[list_rows[j]] == list_ids[j]`.
- list codes are trusted compressed payloads and must be length-consistent with list IDs. The loader must not require raw vector mirror re-encoding to reproduce codes, because FAISS-imported IVF-SQ8 indexes may not own real raw vectors.

---

## Tasks

### Task 1: Export Surface

Files:

- Modify: `src/faiss/ivf_sq8.rs`
- Test: `tests/test_ivf_sq8_sectioned_snapshot.rs`

Steps:

- [x] Add `IvfSq8SectionedExport`.
- [x] Add `IvfSq8Index::export_sectioned_snapshot() -> Result<IvfSq8SectionedExport>`.
- [x] Add export shape test, including quantizer metadata and row mapping.

### Task 2: Sectioned Writer

Files:

- Create: `src/faiss/ivf_sq8_snapshot.rs`
- Modify: `src/faiss/mod.rs`
- Test: `tests/test_ivf_sq8_sectioned_snapshot.rs`

Steps:

- [x] Add section constants and `IVF_SQ8_SECTIONS_SNAPSHOT_VARIANT`.
- [x] Add `IvfSq8SectionedSnapshot::from_index`.
- [x] Implement `AnnSnapshot` writer.
- [x] Re-export symbols from `src/faiss/mod.rs`.
- [x] Add expected-section and manifest-length tests.

### Task 3: Loader

Files:

- Modify: `src/faiss/ivf_sq8.rs`
- Modify: `src/faiss/ivf_sq8_snapshot.rs`
- Test: `tests/test_ivf_sq8_sectioned_snapshot.rs`

Steps:

- [x] Add `IvfSq8Index::from_sectioned_snapshot_export`.
- [x] Validate all layout rules and reject malformed artifacts.
- [x] Add `load_ivf_sq8_index_from_artifact`.
- [x] Add memory-store search roundtrip test.
- [x] Add bitset row-preservation test with non-contiguous explicit IDs.

### Task 4: File Helpers and Compatibility

Files:

- Modify: `src/faiss/ivf_sq8_snapshot.rs`
- Test: `tests/test_ivf_sq8_sectioned_snapshot.rs`

Steps:

- [x] Add `save_ivf_sq8_sectioned_snapshot`.
- [x] Add `load_ivf_sq8_sectioned_snapshot`.
- [x] Add file-store roundtrip test.
- [x] Add corrupt-section tests for bad centroid length, bad list rows, and descriptor length mismatch.
- [x] Confirm existing IVF-SQ8 legacy byte/file tests still pass.

---

## Verification

Run:

```bash
cargo test --test test_ivf_sq8_sectioned_snapshot -- --nocapture
cargo test --lib faiss::ivf_sq8::tests -- --nocapture
cargo test --test test_ivf_flat_sectioned_snapshot -- --nocapture
cargo fmt --all -- --check
```

Expected:

- Sectioned IVF-SQ8 snapshots restore runnable search-equivalent runtimes.
- Bitset filtering uses preserved internal rows.
- Legacy IVF-SQ8 persistence remains unchanged.
- No Lance/HannsDB/pgvector files are modified.
