# HNSW Round 10 Layer-0 Slab Audit Design

## Summary

Round 10 reopens HNSW on a structural memory-layout hypothesis:

- round 9 already removed production search-path profiling leakage and cached batch-4 dispatch
- the next credible remaining gap is not another tiny branch-level cleanup, but the fact that Rust still stores layer-0 neighbors and vectors in separate allocations
- native HNSW keeps layer-0 link data and vector data physically adjacent, which is a better fit for the actual layer-0 search access pattern

Round 10 therefore targets a narrow but structural cut: introduce a co-located layer-0 slab for the `layer0 + L2 + no-filter` hot path, and measure whether that memory-layout change materially improves the real same-schema authority lane.

## Context

Round 9 is now complete and authority-backed. After splitting the unprofiled production fast path and caching batch-4 dispatch, Rust HNSW improved from `750.732` qps to `1845.608` qps at recall `0.9909`, while native measured `10348.740` qps at recall `0.95`. The gap shrank from `11.59x` to `5.61x`, but native still leads by a wide margin.

That result matters for prioritization:

- small search-path implementation cleanups can move the authority lane
- the remaining gap is now too large to justify reopening around another bundle of small, weakly attributable micro-optimizations
- the most credible next hypothesis is the memory locality difference between Rust's split storage and native's per-node contiguous layout at layer 0

## Goals

- Prove or disprove that co-locating layer-0 neighbors and vectors reduces the remaining same-schema gap on the authority lane.
- Keep the round narrowly attributable to memory layout rather than mixing in unrelated search micro-optimizations.
- Preserve correctness, persistence behavior, and existing audit surfaces unless a tracked contract explicitly changes.
- Produce a durable audit artifact that proves the new layout is actually active on the production `layer0 + L2 + no-filter` lane.

## Non-Goals

- No new graph-construction heuristics.
- No accepted-candidate link-list prefetch in this round.
- No visited-table compression in this round.
- No metric-general memory-layout rewrite for upper layers, IP, cosine, or binary paths.
- No full-node single-allocation rework for every HNSW structure in one pass.
- No family verdict refresh claim unless a fresh authority rerun justifies a later tracked feature.

## Problem Statement

Today the production layer-0 L2 fast path still crosses multiple allocations:

- vector data lives in `self.vectors: Vec<f32>`
- layer-0 neighbor ids live in `self.layer0_flat_graph.neighbors: Vec<u32>`
- metadata remains in `self.node_info`

The hot loop therefore does this repeatedly:

1. read a node's layer-0 neighbor ids from the flat graph
2. convert neighbor ids into indices
3. jump to a separate vector allocation to compute distances

Native HNSW uses a very different physical layout: layer-0 links and vector payload are adjacent for each node. That does not make the algorithm different, but it does make the memory access pattern better aligned with hardware prefetch and cache-line reuse.

Round 10 is about testing whether this physical-layout difference is now the dominant remaining Rust-side handicap.

## Proposed Approaches

### Option 1: Dedicated layer-0 slab for the production fast path

Introduce a new slab that stores, per node:

- layer-0 degree
- fixed-capacity layer-0 neighbor ids
- contiguous vector payload

The slab is used only by the production `layer0 + L2 + no-filter` fast path. Existing storage remains authoritative for build logic, upper layers, persistence, and non-L2 fallback behavior.

Pros:

- strongest attribution to memory locality
- minimal interference with upper-layer and non-L2 logic
- preserves the round-9 fast path as the algorithmic shell while swapping only the data source

Cons:

- requires careful synchronization between canonical graph/vector state and the slab
- still touches build, add, load, and refresh paths

### Option 2: Full HNSW node co-location

Replace the current split representation across all layers and paths with a single contiguous node layout.

Pros:

- maximum theoretical alignment with native

Cons:

- too large for a single reopen round
- high persistence and correctness risk
- poor attribution if authority results are mixed

### Option 3: Smaller prefetch-only follow-up

Leave storage split and only add more explicit prefetch operations.

Pros:

- smaller code delta

Cons:

- weak compared with the size of the remaining gap
- does not test the most structurally credible cause

## Recommendation

Choose Option 1.

Round 10 should treat the layer-0 slab as a production-side acceleration structure, not as a whole-index representation rewrite. That keeps the hypothesis focused:

- same search algorithm
- same ordered-pool frontier/result behavior
- same batch-4 distance path
- different physical layout for layer-0 hot data

If this does not move the authority lane, the reopen line gains strong evidence that the remaining gap lies elsewhere. If it does move the lane, later rounds can decide whether broader co-location work is justified.

## Design

### 1. New layer-0 slab structure

Add a dedicated internal structure in `src/faiss/hnsw.rs` for production layer-0 search:

```rust
struct Layer0Slab {
    data: Vec<u8>,
    stride: usize,
    links_size: usize,
    vector_offset: usize,
    max_neighbors: usize,
    dim: usize,
    enabled: bool,
}
```

Per-node layout:

- `u16` or `u32` degree/count
- fixed-capacity `u32` neighbor slots for `m_max0`
- contiguous `f32` vector payload

The exact count type can be chosen during implementation, but the layout must be fixed-stride so pointer math stays simple and branch-light in the hot path.

### 2. Slab refresh model

Round 10 should keep canonical ownership unchanged:

- `self.vectors` remains the canonical vector store
- `self.node_info` remains the canonical graph metadata store
- `self.layer0_flat_graph` remains available for audit compatibility and fallback

The new slab is a derived acceleration structure refreshed whenever layer-0 graph data changes materially, similar in spirit to `refresh_layer0_flat_graph()`.

This keeps the migration risk down:

- build logic can continue to mutate canonical state first
- the slab can be rebuilt after add/load/repair paths
- persistence compatibility is preserved because the slab is rebuildable cache, not serialized source-of-truth state

### 3. Fast-path eligibility

The slab-backed path should only be used when all of these are true:

- metric is `L2`
- no filter is active
- search is on layer 0
- slab is built and enabled for the current node count

Otherwise existing round-9 paths remain in use.

### 4. Hot-loop access model

The production layer-0 fast path should stop bouncing between `layer0_flat_graph.neighbors` and `self.vectors` when slab mode is active.

Instead it should:

1. fetch a candidate node's slab pointer
2. read degree and neighbor ids from the slab
3. compute neighbor vector pointers directly from slab base + offset
4. pass those pointers to the existing scalar/batch-4 L2 kernels

This keeps the algorithm stable while removing one of the biggest remaining locality gaps.

### 5. Audit artifact requirements

Round 10 must not rely on code inspection alone. The audit artifact should prove:

- the production layer-0 fast path uses `layer0_slab` rather than `flat_u32_adjacency`
- the slab stores vectors co-located with layer-0 neighbors
- the profiled path still exists and can report the active layout mode
- the slab is rebuildable from canonical state rather than becoming a silent persistence fork

Suggested fields:

- `production_layer0_layout_mode`
- `profiled_layer0_layout_mode`
- `layer0_slab_enabled`
- `layer0_slab_stride_bytes`
- `layer0_slab_vector_offset_bytes`
- `layer0_slab_max_neighbors`
- `layer0_slab_rebuild_source`

## Data Flow

1. Round-10 activation freezes round-9 authority evidence as the new baseline.
2. A red default-lane contract requires new round-10 baseline, audit, and summary artifacts.
3. Implementation adds a derived layer-0 slab and a slab-backed production fast path.
4. A long-test generator emits a round-10 audit artifact proving the slab-backed mode is active.
5. Authority rerun refreshes:
   - Rust same-schema row
   - native capture
   - round-10 audit artifact
   - round-10 default contract
6. A round-10 authority summary decides whether the slab hypothesis materially moved the real lane.

## Error Handling and Safety

- The slab must be rebuildable and disposable; it is an optimization cache, not a new source of truth.
- Add/load/reset/repair paths must leave the slab either refreshed or explicitly disabled.
- Search correctness must remain unchanged relative to canonical state.
- Save/load compatibility must not depend on serializing the slab.
- If slab rebuild fails or is disabled, production search must fall back safely to the existing round-9 fast path.

## Testing Strategy

### Red phase

- Add a round-10 default-lane contract that fails until baseline, audit, and authority summary artifacts exist.
- Add focused library regressions that fail until:
  - layer-0 slab rebuilds from canonical state
  - the production fast path reports slab layout mode when eligible
  - fallback remains stable when slab mode is unavailable

### Green phase

- Implement the derived slab and the narrow production fast-path swap.
- Refresh only the minimum durable surfaces needed for round-10 auditability.

### Verification

Local prefilters:

- targeted HNSW library tests
- round-10 audit/profile generator
- round-10 default contract
- formatting checks

Authority acceptance:

- `bash init.sh`
- authority round-10 audit/profile replay
- authority same-schema HDF5 rerun
- fresh native HNSW capture
- authority default contract replay
- validator pass on updated durable state

## Expected Outcomes

There are three acceptable round-10 conclusions:

1. Clear authority-lane gain with trustworthy attribution to the layer-0 slab, which justifies a later follow-up around broader memory locality.
2. Little or no gain, which strongly weakens the "split storage is the dominant remaining cause" hypothesis.
3. Mixed result with meaningful qps gain but unacceptable recall or complexity cost, which should be archived honestly rather than spun as a win.

Round 10 is successful if it produces a durable, authority-backed answer about the memory-layout hypothesis, not only if it achieves parity.
