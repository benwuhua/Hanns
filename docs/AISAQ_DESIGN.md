# AISAQ Design

## Scope

This document defines the first Rust skeleton for DiskANN SSD-optimized AISAQ support.
The target is feature parity at the architecture level, not a full port of the existing
C++ `diskann_aisaq.cc` and `pq_flash_aisaq_index.cpp`.

This phase focuses on:

- Core configuration and type model
- PQ Flash index ownership and lifecycle
- Beam-search IO accounting and cache hooks
- A stable storage layout abstraction
- A minimal, compilable `new()`, `train()`, `add()`, and `search()` path

This phase explicitly does not implement:

- Async Linux AIO or io_uring readers
- On-disk persistence and mmap-backed reads
- Reordered vector layouts
- Sample-query driven cache generation
- Full refine/reorder pipelines from C++

## C++ Reference Mapping

### `diskann_aisaq.cc`

The C++ node class is responsible for:

- Translating knowhere config into AISAQ search/build parameters
- Preparing required files and optional files
- Loading `PQFlashAisaqIndex`
- Routing search calls into `aisaq_cached_beam_search`
- Managing cache sizing and file-manager integration

Rust phase-1 keeps this layer in-process and collapses file-manager concerns into
index-owned state.

### `pq_flash_aisaq_index.cpp`

The C++ core contributes four major ideas that shape the Rust skeleton:

- Flash node layout: coordinates, neighborhood list, and inline PQ payload
- Multiple entry points and medoid-style bootstrap
- PQ-aware beam search with IO budgeting
- Separate caches for graph nodes and PQ vectors

The Rust implementation preserves these concepts through explicit data structures,
but stores the first version fully in memory while tracking future SSD behavior
through `BeamSearchIO` and `FlashLayout`.

## Rust Module Layout

New module:

- `src/faiss/diskann_aisaq.rs`

Public types:

- `AisaqConfig`
- `PQFlashIndex`
- `BeamSearchIO`
- `BeamSearchStats`
- `FlashLayout`

Existing `src/faiss/aisaq.rs` remains untouched and is treated as a separate,
older in-memory AISAQ prototype.

## Data Structures

### `AisaqConfig`

`AisaqConfig` merges the existing Rust DiskANN-style settings with the AISAQ-specific
fields found in C++:

- `max_degree`
- `search_list_size`
- `beamwidth`
- `vectors_beamwidth`
- `disk_pq_dims`
- `pq_code_budget_gb`
- `build_dram_budget_gb`
- `pq_cache_size`
- `pq_read_page_cache_size`
- `rearrange`
- `inline_pq`
- `num_entry_points`
- `warm_up`
- `filter_threshold`

Design notes:

- The config is intentionally storage-aware, even if phase-1 uses in-memory backing.
- `vectors_beamwidth` is tracked separately from graph `beamwidth` so later PQ IO can be
  tuned independently.
- Cache sizes are expressed in bytes to match the C++ API surface.

### `FlashLayout`

`FlashLayout` describes the logical node layout that will eventually back SSD pages:

- `page_size`
- `vector_bytes`
- `inline_pq_bytes`
- `neighbor_bytes`
- `node_bytes`

This abstraction decouples search logic from physical storage. In phase-1 it is used
for accounting and tests, not persistence.

### `FlashNode`

Internal node representation:

- `id: i64`
- `vector_offset: usize`
- `neighbors: Vec<u32>`
- `inline_pq: Vec<u8>`

The node object mirrors the C++ disk node composition:

- raw vector payload
- neighborhood list
- optional inline PQ codes

### `PQFlashIndex`

`PQFlashIndex` is the primary index owner. It stores:

- immutable index metadata: metric, dimension, config, flash layout
- raw vector storage: flat `Vec<f32>`
- graph storage: `Vec<FlashNode>`
- PQ state: optional `PqEncoder`, encoded PQ bytes, code size
- search bootstrap state: entry points
- cache state: cached node ids and cached PQ ids
- lifecycle state: `trained`

This aligns with the C++ split between node graph, PQ vectors, and entry-point metadata.

### `BeamSearchIO`

`BeamSearchIO` models the search-time storage subsystem:

- per-query read counters
- page-read accounting
- cache hit/miss tracking
- max node batch / beam read limits

Phase-1 purpose:

- keep search code structured as if it were reading from SSD
- make later async IO integration local to one component
- allow tests to assert cache and read behavior without needing actual files

### `BeamSearchStats`

Per-query stats:

- `nodes_visited`
- `nodes_loaded`
- `node_cache_hits`
- `pq_cache_hits`
- `bytes_read`
- `pages_read`

These stats map cleanly to the C++ IO-centric search path.

## API Design

Primary constructor and methods live on `PQFlashIndex`.

### Construction

```rust
pub fn new(config: AisaqConfig, metric_type: MetricType, dim: usize) -> Result<Self>
```

Responsibilities:

- validate dimension and key config values
- derive `FlashLayout`
- initialize empty graph, caches, and IO manager

### Training

```rust
pub fn train(&mut self, training_data: &[f32]) -> Result<()>
```

Phase-1 training only trains PQ if configured.

Behavior:

- validate input length against `dim`
- create and train `PqEncoder` when `disk_pq_dims > 0`
- mark the index trained even if PQ is disabled

### Ingestion

```rust
pub fn add(&mut self, vectors: &[f32]) -> Result<()>
```

Behavior:

- auto-train on first add if needed
- append raw vectors into flash storage buffer
- encode PQ bytes if a quantizer exists
- create `FlashNode` records
- connect nodes with a simplified Vamana-style neighborhood rule
- refresh entry points

### Search

```rust
pub fn search(&self, query: &[f32], k: usize) -> Result<SearchResult>
```

Behavior:

- validate trained state and query dimension
- initialize per-query `BeamSearchIO`
- seed beam search from entry points
- expand graph candidates while accounting for simulated flash reads
- use PQ distance for coarse candidate ordering when available
- rerank final top-k by exact distance

## Storage Format

Phase-1 defines a stable logical format, even though it is still held in RAM.

### Logical Node Layout

```text
+-------------------+
| vector[f32; dim]  |
+-------------------+
| degree: u32       |
| neighbors[u32; R] |
+-------------------+
| inline_pq[u8; M]  |
+-------------------+
```

Notes:

- `R` is bounded by `max_degree`
- `M` is bounded by `inline_pq`
- in-memory storage uses variable-length `Vec`, while `FlashLayout` captures the future
  packed layout

### Future File Groups

To stay aligned with C++ naming and responsibility, the future on-disk implementation
should split metadata into:

- graph pages
- PQ pivots / codebooks
- compressed vectors
- optional rearranged vectors
- entry point metadata
- cache seed list

The phase-1 code keeps these concerns as separate fields to avoid a future rewrite.

## Beam Search Algorithm

### Search Stages

1. Build a PQ distance table for the query if PQ is enabled.
2. Seed the frontier with `entry_points`.
3. Repeatedly pop the best candidate from the beam.
4. Read its node through `BeamSearchIO`.
5. Push unvisited neighbors using coarse distance:
   - PQ distance if available
   - exact distance otherwise
6. Track the best exact candidates in a bounded result heap.
7. Stop when:
   - the search list budget is exhausted, or
   - the frontier is empty.
8. Rerank the retained candidates with exact distance and emit top-k.

### Why This Matches C++ Well Enough

- Entry points mirror medoid bootstrapping.
- `BeamSearchIO` preserves the IO-shaped search loop.
- Distinguishing coarse PQ scoring from final exact rerank matches `PQFlashAisaqIndex`
  structure without requiring full SSD readers yet.

## Caching Strategy

Two cache classes are modeled now:

- Node cache: graph nodes already resident in memory
- PQ cache: compressed vector payloads already resident in memory

Phase-1 implementation keeps both as explicit id sets. That is sufficient for:

- search-path branching
- stats collection
- later replacement with LRU or page cache objects

## Risks And Deferred Work

- Search quality is limited by the simplified graph construction.
- No persistence means node layout is validated logically, not physically.
- No async IO means beamwidth currently controls expansion, not actual outstanding reads.
- No reorder/refine data path means the current search is closer to a PQ-assisted graph
  scan than the complete C++ AISAQ implementation.

## Implementation Plan

Phase-1:

- add design doc
- add new `diskann_aisaq.rs`
- expose core structs
- compile and test minimal search flow

Phase-2:

- introduce serialization and file groups
- add page cache and mmap-backed reads
- implement PQ vector cache and page cache separation
- add reorder-data refine path

Phase-3:

- async AIO/io_uring reader abstraction
- sample-query driven cache generation
- parity benchmarking against C++ DiskANN AISAQ
