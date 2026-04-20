# Hanns ANN Library and Kernel Storage ABI Design

> Status: design draft
> Date: 2026-04-20
> Scope: Hanns core architecture, plus Lance, pgvector, and HannsDB integration paths

## Goal

Hanns should remain an ergonomic ANN library while also becoming an embeddable ANN kernel for storage engines. The library surface keeps the current `IndexConfig`, `SearchRequest`, `save/load`, and index-specific types. The kernel surface exposes low-allocation search runtimes, typed snapshot sections, and host-provided artifact stores so Lance, pgvector, and HannsDB can integrate without treating Hanns indexes as opaque blobs.

## Non-Goals

- Replacing Lance, Postgres, or HannsDB storage managers.
- Forcing every index family into one physical format.
- Removing current `save/load` or `serialize_to_bytes` compatibility APIs.
- Implementing online DiskANN and pgvector page-native insertion in the first phase.

## Current Problem

Today Hanns is mostly shaped as:

```text
build in memory -> serialize blob -> deserialize blob -> search
```

That is acceptable for simple library users, but it is the wrong primitive for storage engines:

- Lance already owns index metadata, object storage, IVF partitions, and auxiliary files.
- pgvector/Postgres owns buffer pages, WAL, relcache invalidation, vacuum, and heap TID visibility.
- HannsDB owns segments, forward stores, tombstones, and collection-level runtime caches.

The deeper integration shape should be:

```text
host storage -> open typed snapshot sections -> cache runtime once -> search hot path
```

## Target Architecture

```text
                         +----------------------+
                         |        Hanns         |
                         +----------+-----------+
                                    |
             +----------------------+----------------------+
             |                                             |
      +------v------+                              +-------v------+
      | ANN Library |                              | ANN Kernel   |
      | ergonomic   |                              | embeddable   |
      +------+------+                              +-------+------+
             |                                             |
             |                               +-------------v-------------+
             |                               | Runtime / Snapshot / Store|
             |                               +-------------+-------------+
             |                                             |
      +------v------+          +-------------+-------------+-------------+
      | Default     |          |             |                           |
      | mem/file    |    +-----v-----+ +-----v------+             +------v------+
      | stores      |    | HannsDB   | | Lance      |             | pgvector    |
      +-------------+    | segments  | | index store|             | PG pages    |
                         +-----------+ +------------+             +-------------+
```

### Layer Responsibilities

Library layer:

- Owns simple user-facing APIs.
- May allocate, copy, and own `Vec` buffers.
- Provides stable compatibility methods: `save`, `load`, `serialize_to_bytes`, `deserialize_from_bytes`.
- Uses default in-memory/file-backed artifact stores internally.

Kernel layer:

- Owns hot query execution.
- Uses caller-provided output buffers.
- Avoids per-query deserialize and avoidable heap allocation.
- Accepts host storage readers for file, mmap, object-store, and page-backed layouts.
- Provides index-family specific runtimes: graph, IVF, quantized scan, and DiskANN page graph.

## Core Interfaces

```rust
pub trait AnnRuntime: Send + Sync {
    fn family(&self) -> IndexFamily;
    fn dim(&self) -> usize;
    fn len(&self) -> usize;

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
        filter: &dyn RowFilter,
        ids: &mut [i64],
        dists: &mut [f32],
    ) -> Result<usize>;
}
```

```rust
pub trait AnnSnapshot {
    fn manifest(&self) -> &IndexManifest;
    fn write_to(&self, writer: &mut dyn IndexArtifactWriter) -> Result<()>;
}
```

```rust
pub trait AnnSnapshotLoader {
    fn open_runtime(
        reader: &dyn IndexArtifactReader,
        mode: LoadMode,
    ) -> Result<Box<dyn AnnRuntime>>;
}
```

```rust
pub enum LoadMode {
    OwnedMemory,
    Mmap,
    PageCache,
    Lazy,
}
```

```rust
pub trait IndexArtifactReader {
    fn read_section(&self, name: &str) -> Result<Cow<'_, [u8]>>;
    fn section_len(&self, name: &str) -> Result<u64>;
    fn read_range(&self, name: &str, offset: u64, len: usize) -> Result<Vec<u8>>;
}

pub trait IndexArtifactWriter {
    fn write_section(&mut self, name: &str, bytes: &[u8]) -> Result<()>;
    fn finish_manifest(&mut self, manifest: &IndexManifest) -> Result<()>;
}
```

The exact Rust signatures can evolve, but the boundary is stable: runtime searches, snapshot persists, store reads and writes.

## Unified Manifest

Every snapshot has a manifest:

```text
manifest
  magic
  version
  index_family
  index_variant
  dim
  metric
  count
  id_width
  vector_encoding
  sections[]
  checksums[]
```

Common section kinds:

```text
ids
deleted.bitmap
raw_vectors
encoded_vectors
quantizer.model
ivf.centroids
ivf.lists.offsets
ivf.lists.row_ids
ivf.lists.codes
ivf.lists.raw_vectors
graph.levels
graph.neighbors.offsets
graph.neighbors.ids
graph.neighbors.dists
diskann.nodes
diskann.node_offsets
diskann.entry_points
diskann.pq_codes
diskann.raw_vector_pages
```

## Index Families

### HNSW / Graph Family

HNSW should keep the library API but gain a graph runtime over sectioned storage:

```text
hnsw snapshot
  meta
  ids
  raw_vectors or encoded_vectors
  graph.levels
  graph.neighbors.offsets
  graph.neighbors.ids
  graph.neighbors.dists optional
  deleted.bitmap optional
  layer0.slab optional, rebuildable
```

The hot runtime should search over a graph view:

```rust
pub trait GraphView {
    fn entry_point(&self) -> Option<u32>;
    fn level(&self, node: u32) -> u16;
    fn neighbors(&self, node: u32, level: u16) -> &[u32];
    fn vector(&self, node: u32) -> VectorRef<'_>;
}
```

The current `Vec<NodeInfo>` object remains valid for the library facade, but the storage-native runtime should prefer CSR-like sections.

### IVF Family

IVF should be represented as partitioned storage:

```text
ivf snapshot
  meta
  ids
  ivf.centroids
  ivf.lists.offsets
  ivf.lists.row_ids
  ivf.lists.codes or ivf.lists.raw_vectors
  quantizer.model optional
  per-list subindex optional
```

Runtime flow:

```text
query
  -> compute centroid distances
  -> choose nprobe lists
  -> scan raw vectors or encoded codes
  -> optional rerank
  -> map internal ids to external ids
```

IVF list IDs and offsets are storage boundaries. Lance can map them to partitions and auxiliary files. pgvector can map them to page chains.

### Quantization Layer

Quantization is a reusable encoding layer, not a separate storage engine:

```text
model sections:
  SQ: min, max, scale, bit width
  PQ: m, nbits, codebooks
  OPQ/PCA: projection matrix
  RQ: residual codebooks
  USQ/HVQ: rotation, metadata, scales, codes
```

Interfaces:

```rust
pub trait QuantizerModel: Send + Sync {
    type QueryState;

    fn dim(&self) -> usize;
    fn code_size(&self) -> usize;
    fn precompute_query(&self, query: &[f32]) -> Self::QueryState;
    fn score_code(&self, state: &Self::QueryState, code: &[u8]) -> f32;
}

pub trait EncodedVectorStore {
    fn code(&self, row: usize) -> &[u8];
    fn code_range(&self, range: Range<usize>) -> CodeBlock<'_>;
}

pub trait RerankStore {
    fn raw_vector(&self, row: usize) -> Option<&[f32]>;
}
```

`QueryState` is per-query and not persisted. `QuantizerModel` and encoded code sections are persisted.

### DiskANN / Page Graph Family

DiskANN should not be forced into full materialization. It needs a page-oriented runtime:

```text
diskann snapshot
  meta
  ids
  diskann.entry_points
  diskann.node_offsets
  diskann.nodes
  diskann.pq_codes
  diskann.raw_vector_pages optional
  quantizer.model
  deleted.bitmap optional
```

Runtime flow:

```text
query
  -> precompute quantized query state
  -> beam graph traversal
  -> read node records through NodeReader
  -> approximate score
  -> optional raw-vector rerank
  -> top-k
```

Node reader:

```rust
pub trait NodeReader: Send + Sync {
    fn read_node(&self, node_id: u32) -> Result<NodeRecord<'_>>;
    fn prefetch_nodes(&self, node_ids: &[u32]);
}
```

Load modes matter most for DiskANN:

```text
OwnedMemory: small/local memory path
Mmap: local SSD path
PageCache: Postgres or custom page cache
Lazy: remote/object-store path
```

## Library Compatibility

Existing methods should remain:

```rust
index.save(path)
Index::load(path)
index.serialize_to_bytes()
Index::deserialize_from_bytes(bytes)
```

Internally:

```text
save
  -> create snapshot
  -> write to FileArtifactStore

load
  -> open FileArtifactStore
  -> open runtime
  -> wrap runtime in library facade

serialize_to_bytes
  -> write snapshot to MemoryArtifactBundle

deserialize_from_bytes
  -> open MemoryArtifactBundle
  -> open runtime
```

This preserves library ergonomics while moving storage engines onto the kernel path.

## Integration Plans

### HannsDB

HannsDB can adopt the new storage ABI first because it already has segments and an ANN runtime cache.

Target layout:

```text
collections/<name>/
  segments/
  ann/<field>/
    manifest
    ids
    graph / ivf / quantizer / diskann sections
```

Flow:

```text
optimize
  -> load live rows from segment/forward-store
  -> build runtime
  -> persist snapshot
  -> cache runtime

open
  -> open snapshot once
  -> cache runtime

insert/delete
  -> invalidate runtime or update tombstone
```

### Lance

Lance should integrate the kernel directly, not the library facade.

Short-term fix:

```text
load
  -> deserialize Hanns HNSW once
  -> keep Arc runtime

search
  -> use cached runtime
```

Medium-term layout:

```text
_indices/<uuid>/
  index.idx
    IVF metadata
    partition metadata
    subindex metadata

  auxiliary.idx
    ids
    encoded vectors
    quantizer model
    optional graph sections
```

Lance IVF + quantization flow:

```text
build
  -> Lance trains/owns IVF partitioning
  -> Hanns trains quantizer or accepts Lance-provided model
  -> Hanns encodes vectors and writes list sections
  -> Lance writes metadata and object-store artifacts

search
  -> Lance selects candidate partitions
  -> Hanns scans list codes or local subindex
  -> Lance applies prefilter and maps row ids
```

### pgvector

pgvector should treat Postgres pages as the authoritative snapshot store and Hanns runtime as an acceleration cache.

Page layout:

```text
Block 0: MetaPage
  magic
  version
  index_family
  dim
  metric
  count
  section directory root
  cache epoch

Section directory pages:
  section kind
  first block
  block count
  logical length
  checksum

Data pages:
  ids
  centroids
  IVF lists
  quantizer model
  graph
  vectors/codes
  tombstone
```

Phases:

```text
P1: full build writes page-native sections
P2: cache miss opens sections without building one blob Vec<u8>
P3: aminsert writes pending list; query merges ANN + pending brute force
P4: merge pending into IVF/graph
P5: vacuum tombstone and compaction
```

## Rollout

1. Add core traits and in-memory artifact store.
2. Add file artifact store and manifest format.
3. Implement HNSW runtime wrapper over current object representation.
4. Add HNSW sectioned snapshot writer/reader.
5. Add IVF snapshot and list scanner surfaces.
6. Add quantizer model/store surfaces for SQ/PQ/USQ/HVQ.
7. Adapt DiskANN `FileGroup/PageCache` to `IndexArtifactReader` and `NodeReader`.
8. Migrate HannsDB to v2 snapshots.
9. Fix Lance per-query deserialize; then migrate to sectioned artifacts.
10. Replace pgvector blob pages with section directory pages.

## Acceptance Criteria

Core:

- Existing library save/load and byte roundtrips remain compatible.
- New `AnnRuntime::search_into` is covered by tests for HNSW and at least one IVF family.
- Memory artifact bundle roundtrips without filesystem access.
- File artifact store roundtrips through manifest + sections.

HannsDB:

- Optimize writes v2 snapshot.
- Reopen loads v2 snapshot without rebuilding.
- v1 blob fallback still works.

Lance:

- No per-query Hanns HNSW deserialize in the hot search path.
- IVF + quantized list scan can use Hanns kernel without opaque blob storage.

pgvector:

- Initial page-native build can reopen and search.
- Relcache invalidation reloads runtime from page sections.
- Future pending insert path is explicitly represented in the storage ABI.

## Risks

- Snapshot ABI churn can break downstream consumers if versioning is weak.
- Trying to convert all index families at once could stall delivery.
- pgvector page-native DiskANN is a large project and should follow after IVF/HNSW sectioning.
- Lance remote object-store behavior needs careful IO budgeting; full materialization would hide correctness issues in local tests.

## Recommended First Cut

Implement the kernel/storage ABI as additive scaffolding first:

- `src/kernel/` with runtime/filter traits.
- `src/storage/` with manifest and memory/file artifact stores.
- HNSW wrapper runtime using existing `HnswIndex`.
- HNSW v2 snapshot only after the trait boundary lands.

This keeps the first diff reviewable and preserves existing behavior while creating the extension points needed by Lance, pgvector, and HannsDB.
