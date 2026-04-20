# Hanns Core Sectioned Snapshot Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor Hanns core persistence from HNSW opaque blob compatibility toward a sectioned storage ABI that can later support Lance, pgvector, HannsDB, IVF, quantization, and DiskANN without losing the current ANN library API.

**Architecture:** Keep current `HnswIndex::save/load/serialize_to_bytes/deserialize_from_bytes` behavior intact while adding a new HNSW sectioned snapshot path behind the existing `AnnSnapshot` / `AnnSnapshotLoader` / `IndexArtifactReader` boundaries. The first sectioned format materializes back into `HnswIndex` for search, so the runtime behavior stays stable while storage layout becomes inspectable and host-adaptable.

**Tech Stack:** Rust 2021, existing `src/faiss/hnsw.rs`, `src/faiss/hnsw_snapshot.rs`, `src/kernel/*`, `src/storage/*`, `serde`, file/memory artifact stores, existing cargo test/fmt gates.

---

## Scope

This plan is Hanns-core only. Do not edit Lance, HannsDB, or pgvector repositories while executing it.

Current branch state:

- `src/kernel/*` has additive runtime contracts for HNSW, IVF, quantization, and DiskANN.
- `src/storage/*` has `IndexManifest`, `MemoryArtifactStore`, `FileArtifactStore`, `AnnSnapshot`, and `AnnSnapshotLoader`.
- `src/faiss/hnsw_runtime.rs` wraps existing `HnswIndex` as `AnnRuntime`.
- `src/faiss/hnsw_snapshot.rs` currently stores `hnsw.bytes` as a compatibility blob variant: `hnsw_blob_v1`.

Target state for this plan:

- Add `hnsw_sections_v1` as a second HNSW snapshot variant.
- Keep `hnsw_blob_v1` readable and writable for compatibility.
- Add tests proving blob and sectioned snapshots produce equivalent search results.
- Avoid rewriting `HnswIndex` internals in the first pass; expose only the minimal read-only snapshot export/import hooks needed for a sectioned snapshot.

---

## File Structure

Modify:

- `src/faiss/hnsw.rs`
  - Add narrow read-only export helpers and sectioned import constructor for HNSW persistence.
  - Do not change existing `write_to`, `read_from`, `save`, `load`, or byte serialization semantics.

- `src/faiss/hnsw_snapshot.rs`
  - Add `HnswSectionedSnapshot`.
  - Add `HnswSnapshotFormat` or constants for `hnsw_blob_v1` and `hnsw_sections_v1`.
  - Extend loader to dispatch by manifest variant.

- `src/storage/manifest.rs`
  - Add optional metadata map only if required for HNSW sections.
  - Prefer section descriptors over ad-hoc metadata when possible.

- `src/faiss/mod.rs`
  - Export new snapshot types/constants as needed.

Create:

- `tests/test_hnsw_sectioned_snapshot.rs`
  - Roundtrip and compatibility coverage for sectioned HNSW snapshots.

Do not modify:

- Lance repository files.
- Existing FFI/JNI behavior except through preserved public Hanns API.
- DiskANN/IVF/quantizer implementations in this first HNSW sectioning pass.

---

## Section Layout v1

Use these section names for `hnsw_sections_v1`:

```text
hnsw.meta.json
hnsw.vectors.f32
hnsw.ids.i64
hnsw.levels.u32
hnsw.neighbors.offsets.u64
hnsw.neighbors.ids.i64
hnsw.neighbors.dists.f32
hnsw.deleted.ids.i64
hnsw.sq.codes.u8
hnsw.sq.meta.json
```

Rules:

- `hnsw.meta.json` contains version, dim, metric, m, m_max0, ef_search, ef_construction, max_level, level_multiplier, count, entry_point, and SQ mode tag.
- `hnsw.vectors.f32` stores current normalized/runtime vector values exactly as `HnswIndex` would serialize them.
- `hnsw.ids.i64` preserves external IDs and sequential/non-sequential behavior.
- Neighbor sections use CSR:
  - offsets length = total logical layer records + 1
  - ids/dists length = total neighbors across all node layers
  - layer records are ordered by node index, then layer index.
- `hnsw.levels.u32` stores `max_layer` per node.
- `hnsw.deleted.ids.i64` may be empty.
- SQ sections are present only when needed; otherwise descriptor may omit them.

Rationale: this layout is not the final mmap CSR runtime. It is the first inspectable, host-adaptable persistence layout that can still materialize into today’s `HnswIndex`.

---

## Chunk 1: HNSW Export Surface

### Task 1: Add Read-Only HNSW Snapshot Export Helpers

**Files:**

- Modify: `src/faiss/hnsw.rs`
- Test: `tests/test_hnsw_sectioned_snapshot.rs`

- [ ] **Step 1: Write failing export-shape test**

Create `tests/test_hnsw_sectioned_snapshot.rs` with a small HNSW index and an assertion that it can produce a section export:

```rust
#[test]
fn hnsw_exports_sectioned_snapshot_shape() {
    let index = build_small_hnsw();
    let export = index.export_sectioned_snapshot().expect("export");

    assert_eq!(export.dim, 4);
    assert_eq!(export.count, 4);
    assert_eq!(export.ids.len(), 4);
    assert_eq!(export.levels.len(), 4);
    assert_eq!(export.vectors.len(), 16);
    assert_eq!(export.neighbor_offsets.first().copied(), Some(0));
    assert!(export.neighbor_offsets.len() > export.levels.len());
    assert_eq!(export.neighbor_ids.len(), export.neighbor_dists.len());
}
```

Expected RED:

```text
no method named `export_sectioned_snapshot`
```

- [ ] **Step 2: Add `HnswSectionedExport` struct**

In `src/faiss/hnsw.rs`, add a public or `pub(crate)` struct near HNSW persistence code:

```rust
#[derive(Debug, Clone)]
pub struct HnswSectionedExport {
    pub dim: usize,
    pub metric_type: MetricType,
    pub m: usize,
    pub m_max0: usize,
    pub ef_search: usize,
    pub ef_construction: usize,
    pub max_level: usize,
    pub level_multiplier: f32,
    pub entry_point: Option<i64>,
    pub sq_mode: SqMode,
    pub ids: Vec<i64>,
    pub vectors: Vec<f32>,
    pub levels: Vec<u32>,
    pub neighbor_offsets: Vec<u64>,
    pub neighbor_ids: Vec<i64>,
    pub neighbor_dists: Vec<f32>,
    pub deleted_ids: Vec<i64>,
    pub sq_codes: Vec<u8>,
}
```

- [ ] **Step 3: Implement `HnswIndex::export_sectioned_snapshot`**

Implementation requirements:

- Clone vectors and IDs exactly from current runtime state.
- Export each node’s layers in deterministic node/layer order.
- Include an offsets vector that starts with 0 and pushes the cumulative neighbor length after every node layer.
- Export deleted IDs in deterministic order; sort them if current storage is a set.
- Do not change existing `write_to`.

- [ ] **Step 4: Run export-shape test**

Run:

```bash
cargo test --test test_hnsw_sectioned_snapshot hnsw_exports_sectioned_snapshot_shape -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

Use a lore commit message:

```text
Expose HNSW section export without changing persistence

...
Tested: cargo test --test test_hnsw_sectioned_snapshot hnsw_exports_sectioned_snapshot_shape -- --nocapture
Not-tested: sectioned import and loader dispatch
```

---

## Chunk 2: Sectioned Snapshot Writer

### Task 2: Add `HnswSectionedSnapshot` Writer

**Files:**

- Modify: `src/faiss/hnsw_snapshot.rs`
- Modify: `src/faiss/mod.rs`
- Test: `tests/test_hnsw_sectioned_snapshot.rs`

- [ ] **Step 1: Write failing memory-store section writer test**

Add:

```rust
#[test]
fn hnsw_sectioned_snapshot_writes_expected_sections() {
    let index = build_small_hnsw();
    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();

    snapshot.write_snapshot(&mut store).expect("write");
    let manifest = store.manifest().expect("manifest");

    assert_eq!(manifest.family, IndexFamily::Hnsw);
    assert_eq!(manifest.variant, "hnsw_sections_v1");
    assert!(store.section_len("hnsw.meta.json").unwrap() > 0);
    assert_eq!(store.section_len("hnsw.ids.i64").unwrap(), 4 * 8);
    assert_eq!(store.section_len("hnsw.vectors.f32").unwrap(), 16 * 4);
}
```

Expected RED:

```text
use of undeclared type `HnswSectionedSnapshot`
```

- [ ] **Step 2: Implement `HnswSectionedSnapshot::from_index`**

Implementation notes:

- Use `HnswIndex::export_sectioned_snapshot`.
- Build `IndexManifest { family: Hnsw, variant: "hnsw_sections_v1", ... }`.
- Write numeric sections in little-endian byte order.
- Write `hnsw.meta.json` with `serde_json::to_vec_pretty`.
- Reuse `IndexArtifactWriter` and `SectionDescriptor`.

- [ ] **Step 3: Export public symbols**

In `src/faiss/mod.rs`, export:

```rust
HnswSectionedSnapshot
HNSW_SECTIONS_SNAPSHOT_VARIANT
```

- [ ] **Step 4: Run writer test**

Run:

```bash
cargo test --test test_hnsw_sectioned_snapshot hnsw_sectioned_snapshot_writes_expected_sections -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit with `Tested:` trailer.

---

## Chunk 3: Sectioned Import and Loader Dispatch

### Task 3: Load Sectioned Snapshot Back Into `HnswRuntime`

**Files:**

- Modify: `src/faiss/hnsw.rs`
- Modify: `src/faiss/hnsw_snapshot.rs`
- Test: `tests/test_hnsw_sectioned_snapshot.rs`

- [ ] **Step 1: Write failing sectioned roundtrip search test**

Add:

```rust
#[test]
fn hnsw_sectioned_snapshot_roundtrips_search_results() {
    let index = build_small_hnsw();
    let req = request();
    let query = [0.0, 0.0, 0.0, 0.0];
    let expected = index.search(&query, &req).expect("search");

    let snapshot = HnswSectionedSnapshot::from_index(&index).expect("snapshot");
    let mut store = MemoryArtifactStore::default();
    snapshot.write_snapshot(&mut store).expect("write");

    let loader = HnswSnapshotLoader;
    let runtime = loader.load_snapshot(&store, LoadMode::OwnedMemory).expect("load");
    let mut ids = vec![-1; req.top_k];
    let mut dists = vec![f32::INFINITY; req.top_k];
    let n = runtime.search_into(&query, &req, &mut ids, &mut dists).expect("runtime search");

    assert_eq!(&ids[..n], expected.ids.as_slice());
}
```

Expected RED:

```text
invalid HNSW snapshot manifest
```

- [ ] **Step 2: Add `HnswIndex::from_sectioned_snapshot_export`**

Implementation approach:

- Add an import struct mirroring export data or reuse `HnswSectionedExport`.
- Construct a bootstrap `HnswIndex` using `IndexConfig`.
- Populate fields through a dedicated internal method, not by reparsing the old blob.
- Rebuild derived runtime structures:
  - `use_sequential_ids`
  - ID validation set
  - BF16 storage
  - SQ state validation
  - distance dispatch
  - layer0 flat graph
  - trained/config metadata
- Keep all import validation explicit:
  - vectors length = count * dim
  - ids length = count
  - levels length = count
  - offsets are monotonic
  - neighbor ids exist in id set
  - neighbor ids/dists length match
  - entry point exists if count > 0

- [ ] **Step 3: Extend `HnswSnapshotLoader` dispatch**

In `src/faiss/hnsw_snapshot.rs`:

```text
if variant == hnsw_blob_v1:
  existing bytes loader
else if variant == hnsw_sections_v1:
  read sections -> import -> HnswRuntime
else:
  Codec error
```

- [ ] **Step 4: Run roundtrip test**

Run:

```bash
cargo test --test test_hnsw_sectioned_snapshot hnsw_sectioned_snapshot_roundtrips_search_results -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit with `Tested:` trailer.

---

## Chunk 4: Compatibility and File Store Coverage

### Task 4: Prove Blob and Sectioned Snapshots Coexist

**Files:**

- Modify: `tests/test_hnsw_snapshot_bridge.rs`
- Modify: `tests/test_hnsw_sectioned_snapshot.rs`

- [ ] **Step 1: Add blob fallback compatibility test**

Ensure current `hnsw_blob_v1` tests still pass unchanged:

```bash
cargo test --test test_hnsw_snapshot_bridge -- --nocapture
```

- [ ] **Step 2: Add file-store sectioned roundtrip**

Add:

```rust
#[test]
fn hnsw_sectioned_snapshot_roundtrips_through_file_store() {
    let dir = tempfile::tempdir().unwrap();
    let index = build_small_hnsw();
    let snapshot = HnswSectionedSnapshot::from_index(&index).unwrap();
    let mut writer = FileArtifactStore::new(dir.path()).unwrap();
    snapshot.write_snapshot(&mut writer).unwrap();

    let reader = FileArtifactStore::new(dir.path()).unwrap();
    let runtime = HnswSnapshotLoader.load_snapshot(&reader, LoadMode::OwnedMemory).unwrap();
    ...
}
```

- [ ] **Step 3: Add invalid section validation tests**

Cover at least:

- missing `hnsw.meta.json`
- mismatched `ids` length
- non-monotonic offsets

- [ ] **Step 4: Run compatibility tests**

Run:

```bash
cargo test --test test_hnsw_snapshot_bridge -- --nocapture
cargo test --test test_hnsw_sectioned_snapshot -- --nocapture
```

Expected: PASS.

- [ ] **Step 5: Commit**

Commit with `Tested:` trailers.

---

## Chunk 5: Library Facade Bridge

### Task 5: Add Non-Breaking Facade Helpers

**Files:**

- Modify: `src/faiss/hnsw_snapshot.rs`
- Test: `tests/test_hnsw_sectioned_snapshot.rs`

- [ ] **Step 1: Add explicit helper methods**

Add helper APIs without replacing existing save/load:

```rust
impl HnswIndex {
    pub fn save_sectioned_snapshot<P: AsRef<Path>>(&self, root: P) -> Result<()>;
    pub fn load_sectioned_snapshot<P: AsRef<Path>>(root: P) -> Result<Self>;
}
```

If adding methods on `HnswIndex` would clutter `hnsw.rs`, use free functions in `hnsw_snapshot.rs`:

```rust
pub fn save_hnsw_sectioned_snapshot(index: &HnswIndex, root: impl AsRef<Path>) -> Result<()>;
pub fn load_hnsw_sectioned_snapshot(root: impl AsRef<Path>) -> Result<HnswIndex>;
```

Choose the smaller diff.

- [ ] **Step 2: Test file helper roundtrip**

Run:

```bash
cargo test --test test_hnsw_sectioned_snapshot hnsw_sectioned_file_helpers_roundtrip -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Commit**

Commit with `Tested:` trailer.

---

## Chunk 6: Verification

### Task 6: Run Local Hanns Verification

Run:

```bash
cargo fmt --all -- --check
cargo test --test test_hnsw_snapshot_bridge -- --nocapture
cargo test --test test_hnsw_sectioned_snapshot -- --nocapture
cargo test --test test_hnsw_ann_runtime -- --nocapture
cargo test --test test_storage_manifest --test test_memory_artifact_store --test test_file_artifact_store --test test_snapshot_contract -- --nocapture
cargo test --lib --verbose
```

Expected:

- All focused tests pass.
- Library tests pass.

Also run:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

Expected:

- If still blocked by pre-existing all-feature lint/MSRV issues, record the first 10 representative failures and confirm none originate from new sectioned snapshot files.

---

## Follow-Up Plans After HNSW Sectioning

Do not start these until HNSW sectioned snapshots are merged or explicitly approved:

1. **IVF Snapshot Layout**
   - `ivf.centroids`
   - `ivf.lists.offsets`
   - `ivf.lists.row_ids`
   - raw or encoded list payload sections

2. **Quantizer Model Sections**
   - SQ/PQ/USQ/HVQ model section formats
   - query state remains runtime-only
   - encoded vector sections become shared with IVF and DiskANN

3. **DiskANN Artifact Reader**
   - adapt `FileGroup` / `PageCache` to `IndexArtifactReader`
   - introduce `NodeReader` implementation over sections
   - keep io_uring path behind existing feature gates

4. **Downstream Adapters**
   - HannsDB reads sectioned HNSW snapshots.
   - Lance maps sections into `index.idx` / `auxiliary.idx`.
   - pgvector maps sections into page directory + page chains.

---

## Acceptance Criteria

- Current HNSW blob snapshots still load.
- New HNSW sectioned snapshots write all expected sections.
- Sectioned snapshot loader returns a runtime with search results matching the original index.
- File and memory artifact stores both work.
- Existing `HnswIndex::save/load/serialize_to_bytes/deserialize_from_bytes` behavior is unchanged.
- Tests cover invalid/corrupt sectioned artifacts.
- No Lance/HannsDB/pgvector files are modified in this Hanns core plan.

