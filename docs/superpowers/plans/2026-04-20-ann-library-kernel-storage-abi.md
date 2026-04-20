# ANN Library and Kernel Storage ABI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an additive Hanns kernel/storage ABI that preserves the existing ANN library API while enabling storage-native integration for HannsDB, Lance, and pgvector.

**Architecture:** Keep current index implementations as the library facade, then introduce kernel traits, typed artifact stores, and snapshot manifests. The first implementation phase is scaffolding plus HNSW runtime wrapping; IVF, quantization, and DiskANN adopt the same ABI in later phases.

**Tech Stack:** Rust 2021, existing Hanns `api`/`faiss` modules, `serde`, local file IO, existing `cargo build/test/fmt/clippy` gates.

---

## Chunk 1: Core ABI Scaffolding

### Task 1: Add Kernel Runtime Traits

**Files:**
- Create: `src/kernel/mod.rs`
- Create: `src/kernel/runtime.rs`
- Create: `src/kernel/filter.rs`
- Modify: `src/lib.rs`
- Test: `tests/test_kernel_runtime_contract.rs`

- [ ] **Step 1: Write the failing runtime trait test**

Create `tests/test_kernel_runtime_contract.rs`:

```rust
use hanns::api::{MetricType, SearchRequest};
use hanns::kernel::{AnnRuntime, IndexFamily, RowFilter};

struct EmptyRuntime;

impl AnnRuntime for EmptyRuntime {
    fn family(&self) -> IndexFamily {
        IndexFamily::Flat
    }

    fn dim(&self) -> usize {
        4
    }

    fn len(&self) -> usize {
        0
    }

    fn search_into(
        &self,
        _query: &[f32],
        _req: &SearchRequest,
        _ids: &mut [i64],
        _dists: &mut [f32],
    ) -> hanns::api::Result<usize> {
        Ok(0)
    }
}

#[test]
fn ann_runtime_trait_is_public_and_object_safe() {
    let runtime: Box<dyn AnnRuntime> = Box::new(EmptyRuntime);
    assert_eq!(runtime.family(), IndexFamily::Flat);
    assert_eq!(runtime.dim(), 4);
    assert_eq!(runtime.len(), 0);

    let req = SearchRequest {
        top_k: 4,
        nprobe: 8,
        filter: None,
        params: None,
        radius: None,
    };
    let mut ids = [-1_i64; 4];
    let mut dists = [f32::INFINITY; 4];
    assert_eq!(runtime.search_into(&[0.0; 4], &req, &mut ids, &mut dists).unwrap(), 0);
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test --test test_kernel_runtime_contract -- --nocapture`

Expected: FAIL because `hanns::kernel` does not exist.

- [ ] **Step 3: Implement `src/kernel/filter.rs`**

```rust
pub trait RowFilter: Send + Sync {
    fn is_deleted(&self, row: usize) -> bool;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NoFilter;

impl RowFilter for NoFilter {
    #[inline]
    fn is_deleted(&self, _row: usize) -> bool {
        false
    }
}
```

- [ ] **Step 4: Implement `src/kernel/runtime.rs`**

```rust
use crate::api::{Result, SearchRequest};

use super::filter::{NoFilter, RowFilter};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFamily {
    Flat,
    Hnsw,
    Ivf,
    Quantized,
    DiskAnn,
    Sparse,
}

pub trait AnnRuntime: Send + Sync {
    fn family(&self) -> IndexFamily;
    fn dim(&self) -> usize;
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn search_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize>;

    fn search_with_filter_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        _filter: &dyn RowFilter,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        self.search_into(query, req, ids, dists)
    }

    fn search_without_filter_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        self.search_with_filter_into(query, req, &NoFilter, ids, dists)
    }
}
```

- [ ] **Step 5: Add module exports**

`src/kernel/mod.rs`:

```rust
pub mod filter;
pub mod runtime;

pub use filter::{NoFilter, RowFilter};
pub use runtime::{AnnRuntime, IndexFamily};
```

`src/lib.rs`:

```rust
pub mod kernel;
```

- [ ] **Step 6: Run the test**

Run: `cargo test --test test_kernel_runtime_contract -- --nocapture`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/kernel src/lib.rs tests/test_kernel_runtime_contract.rs
git commit -m "Introduce storage-engine runtime boundary

Hanns needs a kernel-level runtime interface that storage engines can
call without going through blob serialization or library-facing index
facades. This adds the additive trait boundary first, without changing
existing index behavior.

Confidence: high
Scope-risk: narrow
Reversibility: clean
Tested: cargo test --test test_kernel_runtime_contract -- --nocapture
Not-tested: downstream Lance or pgvector integration
"
```

## Chunk 2: Artifact Store and Manifest

### Task 2: Add Snapshot Manifest Types

**Files:**
- Create: `src/storage/mod.rs`
- Create: `src/storage/manifest.rs`
- Modify: `src/lib.rs`
- Test: `tests/test_storage_manifest.rs`

- [ ] **Step 1: Write the manifest serialization test**

Create `tests/test_storage_manifest.rs`:

```rust
use hanns::kernel::IndexFamily;
use hanns::storage::{IndexManifest, SectionDescriptor};

#[test]
fn manifest_roundtrips_as_json() {
    let manifest = IndexManifest {
        version: 1,
        family: IndexFamily::Hnsw,
        variant: "hnsw".to_string(),
        dim: 128,
        metric: "l2".to_string(),
        count: 10,
        sections: vec![SectionDescriptor {
            name: "ids".to_string(),
            len: 80,
            checksum: None,
        }],
    };

    let json = serde_json::to_string(&manifest).unwrap();
    let loaded: IndexManifest = serde_json::from_str(&json).unwrap();
    assert_eq!(loaded.family, IndexFamily::Hnsw);
    assert_eq!(loaded.sections[0].name, "ids");
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test --test test_storage_manifest -- --nocapture`

Expected: FAIL because `hanns::storage` does not exist and `IndexFamily` does not serialize yet.

- [ ] **Step 3: Add serde derives to `IndexFamily`**

Modify `src/kernel/runtime.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexFamily { ... }
```

- [ ] **Step 4: Implement manifest types**

`src/storage/manifest.rs`:

```rust
use crate::kernel::IndexFamily;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct IndexManifest {
    pub version: u32,
    pub family: IndexFamily,
    pub variant: String,
    pub dim: usize,
    pub metric: String,
    pub count: usize,
    pub sections: Vec<SectionDescriptor>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SectionDescriptor {
    pub name: String,
    pub len: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,
}
```

`src/storage/mod.rs`:

```rust
pub mod manifest;

pub use manifest::{IndexManifest, SectionDescriptor};
```

`src/lib.rs`:

```rust
pub mod storage;
```

- [ ] **Step 5: Run the manifest test**

Run: `cargo test --test test_storage_manifest -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/kernel/runtime.rs src/storage src/lib.rs tests/test_storage_manifest.rs
git commit -m "Define typed ANN snapshot manifest

The storage ABI needs a stable description of snapshot sections before
file, memory, Lance, or Postgres stores can share loading logic. This
adds only the manifest shape and serde roundtrip coverage.

Confidence: high
Scope-risk: narrow
Reversibility: clean
Tested: cargo test --test test_storage_manifest -- --nocapture
Not-tested: binary compatibility with existing save/load blobs
"
```

### Task 3: Add Memory Artifact Store

**Files:**
- Create: `src/storage/artifact.rs`
- Modify: `src/storage/mod.rs`
- Test: `tests/test_memory_artifact_store.rs`

- [ ] **Step 1: Write memory store tests**

Create `tests/test_memory_artifact_store.rs`:

```rust
use hanns::kernel::IndexFamily;
use hanns::storage::{
    IndexArtifactReader, IndexArtifactWriter, IndexManifest, MemoryArtifactStore,
};

fn manifest() -> IndexManifest {
    IndexManifest {
        version: 1,
        family: IndexFamily::Flat,
        variant: "flat".to_string(),
        dim: 4,
        metric: "l2".to_string(),
        count: 1,
        sections: Vec::new(),
    }
}

#[test]
fn memory_artifact_store_reads_written_sections() {
    let mut store = MemoryArtifactStore::default();
    store.write_section("ids", &[1, 2, 3, 4]).unwrap();
    store.finish_manifest(&manifest()).unwrap();

    assert_eq!(store.section_len("ids").unwrap(), 4);
    assert_eq!(&*store.read_section("ids").unwrap(), &[1, 2, 3, 4]);
    assert_eq!(store.read_range("ids", 1, 2).unwrap(), vec![2, 3]);
    assert_eq!(store.manifest().unwrap().family, IndexFamily::Flat);
}
```

- [ ] **Step 2: Run the failing test**

Run: `cargo test --test test_memory_artifact_store -- --nocapture`

Expected: FAIL because artifact store types do not exist.

- [ ] **Step 3: Implement artifact traits and memory store**

`src/storage/artifact.rs`:

```rust
use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::api::{KnowhereError, Result};

use super::IndexManifest;

pub trait IndexArtifactReader {
    fn manifest(&self) -> Result<&IndexManifest>;
    fn read_section(&self, name: &str) -> Result<Cow<'_, [u8]>>;
    fn section_len(&self, name: &str) -> Result<u64>;
    fn read_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>>;
}

pub trait IndexArtifactWriter {
    fn write_section(&mut self, name: &str, bytes: &[u8]) -> Result<()>;
    fn finish_manifest(&mut self, manifest: &IndexManifest) -> Result<()>;
}

#[derive(Debug, Default, Clone)]
pub struct MemoryArtifactStore {
    manifest: Option<IndexManifest>,
    sections: BTreeMap<String, Vec<u8>>,
}

impl IndexArtifactReader for MemoryArtifactStore {
    fn manifest(&self) -> Result<&IndexManifest> {
        self.manifest
            .as_ref()
            .ok_or_else(|| KnowhereError::Codec("missing manifest".to_string()))
    }

    fn read_section(&self, name: &str) -> Result<Cow<'_, [u8]>> {
        self.sections
            .get(name)
            .map(|bytes| Cow::Borrowed(bytes.as_slice()))
            .ok_or_else(|| KnowhereError::Codec(format!("missing section: {name}")))
    }

    fn section_len(&self, name: &str) -> Result<u64> {
        Ok(self.read_section(name)?.len() as u64)
    }

    fn read_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>> {
        let section = self.read_section(name)?;
        let offset = offset as usize;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| KnowhereError::Codec("section range overflow".to_string()))?;
        let bytes = section
            .get(offset..end)
            .ok_or_else(|| KnowhereError::Codec(format!("section range out of bounds: {name}")))?;
        Ok(bytes.to_vec())
    }
}

impl IndexArtifactWriter for MemoryArtifactStore {
    fn write_section(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        self.sections.insert(name.to_string(), bytes.to_vec());
        Ok(())
    }

    fn finish_manifest(&mut self, manifest: &IndexManifest) -> Result<()> {
        self.manifest = Some(manifest.clone());
        Ok(())
    }
}
```

- [ ] **Step 4: Export artifacts**

`src/storage/mod.rs`:

```rust
pub mod artifact;
pub mod manifest;

pub use artifact::{IndexArtifactReader, IndexArtifactWriter, MemoryArtifactStore};
pub use manifest::{IndexManifest, SectionDescriptor};
```

- [ ] **Step 5: Run artifact tests**

Run: `cargo test --test test_memory_artifact_store -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/storage tests/test_memory_artifact_store.rs
git commit -m "Add in-memory artifact store for ANN snapshots

The kernel ABI needs a filesystem-free store for tests and byte-based
compatibility wrappers. This introduces reader/writer traits and the
first concrete store without changing existing index persistence.

Confidence: high
Scope-risk: narrow
Reversibility: clean
Tested: cargo test --test test_memory_artifact_store -- --nocapture
Not-tested: file-backed store and downstream adapters
"
```

## Chunk 3: HNSW Runtime Wrapper

### Task 4: Wrap Existing HNSW as `AnnRuntime`

**Files:**
- Create: `src/faiss/hnsw_runtime.rs`
- Modify: `src/faiss/mod.rs`
- Test: `tests/test_hnsw_ann_runtime.rs`

- [ ] **Step 1: Write runtime parity test**

Create `tests/test_hnsw_ann_runtime.rs`:

```rust
use hanns::api::{IndexConfig, IndexType, MetricType, SearchRequest};
use hanns::faiss::{HnswIndex, HnswRuntime};
use hanns::kernel::{AnnRuntime, IndexFamily};

#[test]
fn hnsw_runtime_matches_hnsw_search() {
    let dim = 4;
    let mut cfg = IndexConfig::new(IndexType::Hnsw, MetricType::L2, dim);
    cfg.params.m = Some(8);
    cfg.params.ef_construction = Some(32);

    let vectors = vec![
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    ];
    let ids = vec![10, 11, 12, 13];
    let query = [0.0, 0.0, 0.0, 0.0];
    let req = SearchRequest {
        top_k: 2,
        nprobe: 16,
        filter: None,
        params: None,
        radius: None,
    };

    let mut index = HnswIndex::new(&cfg).unwrap();
    index.train(&vectors).unwrap();
    index.add(&vectors, Some(&ids)).unwrap();

    let expected = index.search(&query, &req).unwrap();
    let runtime = HnswRuntime::new(index);
    let mut got_ids = [-1_i64; 2];
    let mut got_dists = [f32::INFINITY; 2];
    let n = runtime.search_into(&query, &req, &mut got_ids, &mut got_dists).unwrap();

    assert_eq!(runtime.family(), IndexFamily::Hnsw);
    assert_eq!(n, expected.ids.len());
    assert_eq!(&got_ids[..n], expected.ids.as_slice());
}
```

- [ ] **Step 2: Run failing test**

Run: `cargo test --test test_hnsw_ann_runtime -- --nocapture`

Expected: FAIL because `HnswRuntime` does not exist.

- [ ] **Step 3: Implement wrapper**

`src/faiss/hnsw_runtime.rs`:

```rust
use crate::api::{Result, SearchRequest};
use crate::kernel::{AnnRuntime, IndexFamily, RowFilter};

use super::HnswIndex;

pub struct HnswRuntime {
    inner: HnswIndex,
}

impl HnswRuntime {
    pub fn new(inner: HnswIndex) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> HnswIndex {
        self.inner
    }
}

impl AnnRuntime for HnswRuntime {
    fn family(&self) -> IndexFamily {
        IndexFamily::Hnsw
    }

    fn dim(&self) -> usize {
        self.inner.dim()
    }

    fn len(&self) -> usize {
        self.inner.ntotal()
    }

    fn search_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        self.inner.search_into(query, req, ids, dists)
    }

    fn search_with_filter_into(
        &self,
        query: &[f32],
        req: &SearchRequest,
        _filter: &dyn RowFilter,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize> {
        self.search_into(query, req, ids, dists)
    }
}
```

- [ ] **Step 4: Export wrapper**

Modify `src/faiss/mod.rs`:

```rust
pub mod hnsw_runtime;
pub use hnsw_runtime::HnswRuntime;
```

- [ ] **Step 5: Run parity test**

Run: `cargo test --test test_hnsw_ann_runtime -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/faiss/hnsw_runtime.rs src/faiss/mod.rs tests/test_hnsw_ann_runtime.rs
git commit -m "Wrap HNSW as kernel runtime

Storage engines need to call Hanns search without owning library facade
state or invoking serialization. This wraps the existing HNSW index as
the first AnnRuntime while preserving current HNSW behavior.

Confidence: medium
Scope-risk: narrow
Reversibility: clean
Tested: cargo test --test test_hnsw_ann_runtime -- --nocapture
Not-tested: native bitset filter mapping through RowFilter
"
```

## Chunk 4: Expand to IVF, Quantization, and DiskANN

### Task 5: Add Family-Specific Kernel Traits

**Files:**
- Create: `src/kernel/ivf.rs`
- Create: `src/kernel/quant.rs`
- Create: `src/kernel/disk_graph.rs`
- Modify: `src/kernel/mod.rs`
- Test: `tests/test_kernel_family_traits.rs`

- [ ] **Step 1: Write object-safety and type-shape tests**

Run: `cargo test --test test_kernel_family_traits -- --nocapture`

Expected before implementation: FAIL.

- [ ] **Step 2: Add IVF traits**

Define `IvfPartitionSelector`, `IvfListScanner`, and small data structs for selected partitions and list hits.

- [ ] **Step 3: Add quantization traits**

Define `QuantizerModel`, `EncodedVectorStore`, and `RerankStore`. Keep associated `QueryState` generic for concrete kernels; do not force trait objects in the first cut.

- [ ] **Step 4: Add DiskANN traits**

Define `NodeReader`, `NodeRecord`, and `DiskGraphRuntimeConfig`. Keep async/io_uring out of the first trait to avoid forcing async into all users.

- [ ] **Step 5: Run tests**

Run: `cargo test --test test_kernel_family_traits -- --nocapture`

Expected: PASS.

- [ ] **Step 6: Commit**

Use a lore commit message explaining that these are contracts only and intentionally do not migrate implementations yet.

## Chunk 5: Verification

### Task 6: Run Local Verification Gates

**Files:** no source edits.

- [ ] **Step 1: Format check**

Run: `cargo fmt --all -- --check`

Expected: PASS.

- [ ] **Step 2: Unit/library tests**

Run: `cargo test --lib --verbose`

Expected: PASS.

- [ ] **Step 3: Focused integration tests**

Run:

```bash
cargo test --test test_kernel_runtime_contract -- --nocapture
cargo test --test test_storage_manifest -- --nocapture
cargo test --test test_memory_artifact_store -- --nocapture
cargo test --test test_hnsw_ann_runtime -- --nocapture
cargo test --test test_kernel_family_traits -- --nocapture
```

Expected: PASS.

- [ ] **Step 4: Clippy**

Run: `cargo clippy --all-targets --all-features -- -D warnings`

Expected: PASS or document pre-existing feature-gated failures.

- [ ] **Step 5: Final commit if verification required changes**

Commit any verification fixes with a lore message including `Tested:` and `Not-tested:` trailers.

## Follow-Up Plans

These are intentionally separate plans after the ABI scaffold lands:

- HannsDB v2 snapshot migration.
- Lance HNSW no-per-query-deserialize fix and sectioned artifact integration.
- IVF/USQ/PQ snapshot implementation.
- DiskANN `FileGroup/PageCache` adaptation to artifact readers.
- pgvector page-native section directory and pending-list insertion.
