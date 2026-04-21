# ANN Snapshot Registry Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development when subagent capacity is available, otherwise execute directly with the same checkpoint discipline.

**Goal:** Consolidate sectioned snapshot loading behind a registry so host adapters can route by manifest variant instead of hardcoding per-index loader calls.

**Architecture:** Keep `storage` generic. Add `AnnSnapshotRegistry` as a variant-to-loader dispatcher over the existing `AnnSnapshotLoader` trait. In `faiss`, add lightweight `IvfRuntime` adapters and `default_ann_snapshot_registry()` that registers HNSW plus the sectioned IVF formats implemented so far.

---

## Scope

In scope:

- Generic `storage::AnnSnapshotRegistry`.
- IVF runtime wrapper for `IvfFlatIndex`, `IvfSq8Index`, `IvfPqIndex`, and `IvfUsqIndex`.
- Loader structs for sectioned IVF formats.
- `faiss::default_ann_snapshot_registry()`.
- Tests proving registry dispatch and unsupported variant errors.

Out of scope:

- Lance/HannsDB/pgvector adapters.
- Mmap/page-cache/lazy runtime implementations.
- Changing existing per-index helper functions.

---

## Tasks

- [x] Add generic registry in `src/storage/snapshot.rs`.
- [x] Add `IvfRuntime` implementing `AnnRuntime`.
- [x] Add IVF `AnnSnapshotLoader` structs wrapping existing artifact loaders.
- [x] Register HNSW and IVF variants in `default_ann_snapshot_registry`.
- [x] Add registry tests.
- [x] Run focused and full verification.

---

## Verification

Run:

```bash
cargo test --test test_ann_snapshot_registry -- --nocapture
cargo test --test test_snapshot_contract -- --nocapture
cargo test --test test_ivf_flat_sectioned_snapshot -- --nocapture
cargo test --test test_ivf_usq_sectioned_snapshot -- --nocapture
cargo test --lib --verbose
cargo fmt --all -- --check
```
