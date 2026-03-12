# HNSW Round 4 Layer-0 Searcher Parity Design

## Goal

Reopen HNSW for a fourth focused algorithm line whose success criterion is improved Rust QPS on the remote x86 same-schema recall-gated authority lane by making the Rust layer-0 `L2 + no-filter` search core structurally closer to the native knowhere implementation, without changing the current API, FFI, persistence, or historical verdict artifacts.

## Context

Round 3 closed honestly as a `soft_stop`. The authority summary in `benchmark_results/hnsw_reopen_round3_authority_summary.json` recorded:

- Rust same-schema HNSW improved from `521.031` to `553.060` qps
- recall improved from `0.9923` to `0.9943`
- native HNSW also improved to `10792.646` qps
- the native-over-Rust gap only narrowed from `20.2x` to `19.5x`

The refreshed round-3 profile in `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json` still shows `distance_compute` as the dominant bucket at about `35.7ms`, but `visited_ops` and `frontier_ops` together remain a large secondary cost. The profile also attributes about `31.1ms` of the measured distance work to `layer0_query_distance`, making layer-0 search the dominant remaining hotspot.

Direct comparison against the local native source tree at `/Users/ryan/Code/vectorDB/knowhere` tightened the reference target:

- actual native search path is the knowhere-specific `v2_hnsw_searcher` in `thirdparty/faiss/faiss/cppcontrib/knowhere/impl/HnswSearcher.h`
- native layer-0 search uses `NeighborSetDoublePopList` from `thirdparty/faiss/faiss/cppcontrib/knowhere/impl/Neighbor.h`, not a pair of generic heaps
- native level-0 neighbor expansion evaluates distances in batches of 4 with `qdis.distances_batch_4(...)`
- Rust still uses two fresh `BinaryHeap`s per search in `src/faiss/hnsw.rs` and still evaluates `query -> node` distances one neighbor at a time

Therefore round 4 should not be framed as "more distance micro-optimizations." It should answer a narrower structural question: can Rust HNSW improve the authority lane by making its layer-0 search core more like the actual native searcher?

## Decision

Round 4 will target `layer0_searcher_parity`. It will not attempt a full graph layout rewrite and it will not reopen the family or project verdict chain directly. The line exists to answer whether a native-like layer-0 candidate container and batched neighbor distance evaluation can produce a materially better same-schema Rust result.

## Scope

Round 4 includes four tracked features:

1. `hnsw-reopen-round4-activation`
2. `hnsw-layer0-searcher-audit`
3. `hnsw-layer0-searcher-core-rework`
4. `hnsw-round4-authority-same-schema-rerun`

Round 4 does not:

- reopen IVF-PQ or DiskANN work
- rewrite `benchmark_results/hnsw_p3_002_final_verdict.json`
- rewrite `benchmark_results/final_core_path_classification.json`
- rewrite project-level acceptance artifacts
- change the current file format, FFI, or public API

## Native Reference Surface

Round 4 uses two native reference files as the primary structural guide:

- `/Users/ryan/Code/vectorDB/knowhere/thirdparty/faiss/faiss/cppcontrib/knowhere/impl/HnswSearcher.h`
- `/Users/ryan/Code/vectorDB/knowhere/thirdparty/faiss/faiss/cppcontrib/knowhere/impl/Neighbor.h`

The specific reference behaviors to mirror are:

- greedy top-level descent remains separate from level-0 expansion
- level-0 candidates/results are managed by an ordered fixed-capacity container rather than dual heaps
- level-0 neighbor expansion batches `query -> node` distance evaluation in groups of 4
- visited-state reuse remains cheap and query-scoped

Round 4 does not require Rust to copy the native types exactly. It only needs to converge on the same operational shape.

## Rust Baseline Surface

The current Rust search flow to be reworked is:

- `search_single_l2_unfiltered()` in `src/faiss/hnsw.rs`
- `greedy_upper_layer_descent_l2_with_entry_dist_optional_profile()`
- `search_layer_idx_l2_with_optional_profile()`
- `SearchScratch`

The current hot-path mismatches versus native are:

- `SearchScratch` only tracks visited epochs, not reusable frontier/result state
- each layer-0 search allocates fresh `BinaryHeap` containers
- layer-0 distance evaluation remains one-by-one even on the L2 fast path
- adjacency remains `Vec<i64>` ids, so search still pays an id-to-index conversion per hop

The first three mismatches are in scope for round 4. The adjacency layout mismatch is acknowledged but explicitly deferred.

## Architecture

Round 4 keeps three layers separate:

- Historical truth layer: the existing HNSW, core-path, leadership, and production-acceptance verdict artifacts remain unchanged throughout round 4.
- Reopen tracking layer: add round-4-specific artifacts that freeze the round-3 soft-stop baseline, record the native-vs-Rust search-core mismatch, and summarize the new same-schema authority rerun.
- Algorithm layer: constrain implementation changes to `src/faiss/hnsw.rs` and, if needed, a small SIMD helper addition that supports batched `query -> node` L2 evaluation.

## Round 4 Hypothesis

The next honest HNSW hypothesis is:

> The current Rust layer-0 `L2 + no-filter` search core is still materially slower than native because it uses a generic dual-heap search shape and one-by-one neighbor distance evaluation, while native uses a more specialized fixed-capacity ordered candidate structure plus batched distance evaluation.

If this hypothesis is correct, then:

- the synthetic round-4 profile should show lower `layer0_query_distance` and/or lower `frontier_ops`
- the remote same-schema Rust HNSW row should improve materially without recall regression

If this hypothesis is wrong, round 4 should prove it quickly and stop.

## Feature Boundaries

### 1. Round 4 Activation

Freeze round 3 as the historical baseline for this line and add a new round-4 activation artifact that records:

- round 3 ended in `soft_stop`
- the new target is `layer0_searcher_parity`
- the historical HNSW verdict remains `functional-but-not-leading`

This feature claims no performance gain. It only changes the active hypothesis and durable workflow state.

### 2. Layer-0 Searcher Audit

Add a new audit/profiler surface that explicitly records the structural gap between Rust and native:

- native reference path and files
- current Rust search-core shape
- whether layer-0 uses `BinaryHeap` or an ordered pool
- whether `query -> node` batch distance is enabled
- batch call counts or equivalent measurements

This feature exists to make the round-4 core rework measurable and falsifiable.

### 3. Layer-0 Searcher Core Rework

Rework the Rust `L2 + no-filter` layer-0 search core while preserving results and external contracts. Allowed changes include:

- replacing dual `BinaryHeap` usage with a scratch-owned fixed-capacity ordered container
- extending `SearchScratch` so layer-0 frontier/result state is query-scoped and reusable
- adding an internal batch-4 L2 helper for `query -> node` distance evaluation
- tightening the layer-0 inner loop to reduce frontier and visited bookkeeping costs when part of the same search core

Round 4 should avoid:

- changing graph construction semantics
- changing vector storage format on disk
- changing the public API or FFI
- changing the filter-bearing path beyond preserving its existing semantics
- changing the internal adjacency layout from id-based per-layer vectors to a fully flattened neighbor array

### 4. Authority Same-Schema Rerun

Re-run the real recall-gated same-schema remote lane and summarize whether round 4 produced enough improvement to justify opening a later verdict-refresh feature.

Round 4 itself does not rewrite the family or project verdict chain.

## Test Strategy

Round 4 uses three verification layers:

- default-lane artifact and contract tests for round-4 activation, audit/profile, and authority summary
- focused HNSW unit/regression tests proving the new layer-0 searcher preserves search behavior and does not affect the filter path
- remote x86 authority replays, with the same-schema HDF5 compare surface as the acceptance lane

The round-4 contract lane should additionally lock:

- `search_core_shape`
- whether batch distance evaluation is active
- batch call counts or equivalent round-4 profile fields

## New Artifacts

Round 4 should add:

- `benchmark_results/hnsw_reopen_round4_baseline.json`
- `benchmark_results/hnsw_reopen_layer0_searcher_audit_round4.json`
- `benchmark_results/hnsw_reopen_round4_authority_summary.json`

The final round-4 summary artifact must record:

- the round-4 target
- the native reference source files
- the resulting Rust search-core shape
- fresh same-schema Rust and native recall/QPS evidence
- deltas versus the round-3 soft-stop baseline
- whether a later verdict-refresh feature is justified
- whether the next action is `continue`, `soft_stop`, or `hard_stop`

## Stop/Go Rules

### Go

- the round-4 profile shows that layer-0 search got materially cheaper through search-core changes, and
- the remote same-schema recall gate holds while Rust QPS improves materially relative to round 3

### Soft Stop

- synthetic/profile evidence improves clearly, but the authority same-schema lane only moves modestly
- in this case only one new explicit hypothesis is justified; round 4 itself does not auto-continue

### Hard Stop

- recall regresses materially
- or the authority same-schema lane regresses again
- or the new audit/profile shows the search-core shape change did not reduce the actual bottleneck
- or further progress would require changing the current API, FFI, persistence, or file-format contract

## Expected Outcome

At the end of round 4 the repository should be able to say one of three things honestly:

1. layer-0 searcher parity materially improved the authority lane and merits a later verdict-refresh feature
2. layer-0 searcher parity improved synthetic or local signals but not enough to move the authority result
3. layer-0 searcher parity did not pay off and HNSW should stop again or change hypotheses
