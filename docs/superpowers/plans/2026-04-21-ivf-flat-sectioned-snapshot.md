# IVF-Flat Sectioned Snapshot Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a sectioned snapshot path for `IvfFlatIndex` as the first IVF family storage ABI implementation, preserving existing IVF-Flat save/load/byte serialization behavior.

**Architecture:** Mirror the HNSW sectioned snapshot pattern: export a deterministic read-only `IvfFlatSectionedExport`, write `ivf_flat_sections_v1` sections through `IndexArtifactWriter`, then add loader support that materializes back into today’s `IvfFlatIndex` runtime. This establishes the IVF section vocabulary before extending to IVF-SQ8, IVF-USQ, and IVF-PQ.

**Tech Stack:** Rust 2021, `src/faiss/ivf_flat.rs`, existing `src/storage/*`, existing `src/kernel/*`, `serde_json`, memory/file artifact stores.

---

## Scope

This plan is Hanns-core only.

In scope:

- `IvfFlatIndex` sectioned export/write/load.
- Memory and file artifact roundtrips.
- Corrupt section validation.
- Existing `IvfFlatIndex::save/load/serialize_to_bytes/deserialize_from_bytes` compatibility.

Out of scope:

- Deprecated `src/faiss/ivf.rs` scaffold.
- IVF-SQ8, IVF-USQ, IVF-PQ section layouts.
- Lance/HannsDB/pgvector adapter changes.

---

## Section Layout v1

Variant:

```text
ivf_flat_sections_v1
```

Sections:

```text
ivf_flat.meta.json
ivf_flat.centroids.f32
ivf_flat.list_offsets.u64
ivf_flat.list_sizes.u64
ivf_flat.list_ids.i64
ivf_flat.list_vectors.f32
ivf_flat.ids.i64
ivf_flat.vectors.f32
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
```

Rules:

- `centroids.len == nlist * dim`
- `list_offsets.len == nlist`
- `list_sizes.len == nlist`
- `sum(list_sizes) == list_ids.len`
- `list_vectors.len == list_ids.len * dim`
- `ids.len == count`
- `vectors.len == count * dim`
- each `list_offsets[i] + list_sizes[i] <= list_ids.len`
- `trained` must be true for runnable snapshots

---

## Chunk 1: IVF-Flat Export Surface

### Task 1: Add `IvfFlatSectionedExport`

**Files:**

- Modify: `src/faiss/ivf_flat.rs`
- Test: `tests/test_ivf_flat_sectioned_snapshot.rs`

Steps:

- [ ] Write failing test `ivf_flat_exports_sectioned_snapshot_shape`.
- [ ] Add `IvfFlatSectionedExport` with metadata and cloned vectors/lists.
- [ ] Add `IvfFlatIndex::export_sectioned_snapshot() -> Result<IvfFlatSectionedExport>`.
- [ ] Validate export shape in test.
- [ ] Run:

```bash
cargo test --test test_ivf_flat_sectioned_snapshot ivf_flat_exports_sectioned_snapshot_shape -- --nocapture
cargo test --test test_hnsw_sectioned_snapshot -- --nocapture
cargo fmt --all -- --check
```

---

## Chunk 2: IVF-Flat Sectioned Writer

### Task 2: Add `IvfFlatSectionedSnapshot`

**Files:**

- Create or modify: `src/faiss/ivf_flat_snapshot.rs`
- Modify: `src/faiss/mod.rs`
- Test: `tests/test_ivf_flat_sectioned_snapshot.rs`

Steps:

- [ ] Write failing writer test `ivf_flat_sectioned_snapshot_writes_expected_sections`.
- [ ] Add section constants and `IVF_FLAT_SECTIONS_SNAPSHOT_VARIANT`.
- [ ] Add `IvfFlatSectionedSnapshot::from_index`.
- [ ] Implement `AnnSnapshot` writer.
- [ ] Export symbols from `src/faiss/mod.rs`.
- [ ] Run focused tests.

---

## Chunk 3: IVF-Flat Loader

### Task 3: Load Sectioned IVF-Flat Snapshot Into Runtime

**Files:**

- Modify: `src/faiss/ivf_flat.rs`
- Modify: `src/faiss/ivf_flat_snapshot.rs`
- Test: `tests/test_ivf_flat_sectioned_snapshot.rs`

Steps:

- [ ] Write failing roundtrip search test.
- [ ] Add `IvfFlatIndex::from_sectioned_snapshot_export`.
- [ ] Add validation for all layout rules.
- [ ] Add loader helper returning `IvfFlatIndex`.
- [ ] Add memory-store roundtrip test matching IDs and distances.
- [ ] Run focused tests.

---

## Chunk 4: Compatibility and File Helpers

### Task 4: Add File Store Roundtrip and Helpers

**Files:**

- Modify: `src/faiss/ivf_flat_snapshot.rs`
- Test: `tests/test_ivf_flat_sectioned_snapshot.rs`

Steps:

- [ ] Add `save_ivf_flat_sectioned_snapshot`.
- [ ] Add `load_ivf_flat_sectioned_snapshot`.
- [ ] Add file-store roundtrip test.
- [ ] Add corrupt-section tests:
  - mismatched list sizes
  - bad centroid length
  - list offset out of range
- [ ] Confirm existing `IvfFlatIndex::serialize_to_bytes` and `deserialize_from_bytes` tests still pass.

---

## Verification

Run:

```bash
cargo test --test test_ivf_flat_sectioned_snapshot -- --nocapture
cargo test --test test_hnsw_sectioned_snapshot -- --nocapture
cargo test --test test_hnsw_snapshot_bridge -- --nocapture
cargo test --lib --verbose
cargo fmt --all -- --check
```

Clippy:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Expected:

- If clippy remains blocked by known pre-existing lint/MSRV issues, record first representative failures and confirm none are in new IVF-Flat sectioned snapshot files.

---

## Acceptance Criteria

- IVF-Flat sectioned snapshot writes readable sections with exact manifest lengths.
- IVF-Flat sectioned snapshot loads back to search-equivalent runtime.
- File and memory stores both work.
- Corrupt IVF-Flat section artifacts are rejected.
- Existing IVF-Flat byte/file persistence remains unchanged.
- No Lance/HannsDB/pgvector files are modified.

