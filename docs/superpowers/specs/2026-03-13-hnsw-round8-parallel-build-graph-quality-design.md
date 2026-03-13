# HNSW Round 8 Parallel-Build Graph Quality Design

## Goal

Open a new HNSW reopen line whose success criterion is improved remote x86 same-schema recall-gated HNSW throughput by fixing parallel-build graph-quality drift in `src/faiss/hnsw.rs`, without changing API, FFI, persistence, or the archived historical verdict artifacts.

## Context

The current HNSW reopen state is no longer a pure round-4 story:

- round 4 moved Rust same-schema HNSW from `553.060` qps to `819.471` qps at recall `0.9959`, but native still reached `12487.076` qps, so the historical family verdict stayed `functional-but-not-leading`
- round 5 cached metric/L2 dispatch correctly, but the same-schema rerun stayed at `819.471` qps and the follow-up stability gate marked ratio movement non-attributable because native swung from `9342.693` to `12808.330` qps
- round 6 and round 7 tightened the search path further with layer-0 prefetch telemetry and flat `u32` adjacency, but those rounds intentionally stopped at audit artifacts rather than claiming a fresh authority move

Those search-side rounds matter because they reduce the likelihood that the remaining gap is dominated by the old layer-0 search-core shape. The repo now has:

- ordered-pool layer-0 search instead of dual heaps
- batch-4 query-distance helpers
- once-per-process single-vector L2 dispatch caching
- prefetch telemetry and vector prefetch in the ordered-pool loop
- flat layer-0 `u32` adjacency for the L2 ordered-pool path

Yet the baseline methodology artifact still says Rust HNSW needs about `ef=2000` to match native recall where native reaches the gate around `ef=139`. That is a graph-quality signal, not a remaining search-loop-only signal.

## Problem

Recent source review against the current Rust code and the native HNSW insertion path points to a narrower build-quality hypothesis than the earlier search-side rounds:

1. The parallel bulk-build helper `find_neighbors_for_insertion_with_scratch()` starts from the global entry point and searches directly from `node_level` down to `0`, instead of first performing greedy descent from `self.max_level` down to `node_level + 1`.
2. The same parallel build path still truncates upper-layer overflow lists with `truncate_to_best()` inside `add_connections_for_node()`, even though the serial insertion path already uses heuristic shrink and the native path uses diversity pruning on overflow.

This means the repo currently has two different insertion-quality surfaces:

- serial insertion: top-down descent plus heuristic shrink
- parallel build: no upper-layer descent and simple upper-layer truncation

That mismatch is large enough to plausibly explain why search-path alignment has not translated into native-comparable recall efficiency.

## Decision

Round 8 will target `parallel_build_graph_quality_parity`.

This line is intentionally narrower than a general "build rewrite." It only asks whether aligning the parallel build path with the already-better serial/native insertion semantics can materially improve same-schema authority evidence.

Round 8 will not bundle the remaining search-side micro-optimizations that were raised during review:

- batch-4 dispatch caching is deferred until after the graph-quality rerun because it is a small search-path optimization and would blur attribution
- AVX2 FMA specialization is also deferred because the current single-vector AVX2 dispatch supports non-FMA hosts; doing it correctly requires a new kernel plus dispatch split, not a one-line intrinsic swap

## Scope

Round 8 includes four tracked features:

1. `hnsw-reopen-round8-activation`
2. `hnsw-parallel-build-graph-audit-round8`
3. `hnsw-parallel-build-graph-rework-round8`
4. `hnsw-round8-authority-same-schema-rerun`

Round 8 does not:

- reopen IVF-PQ, DiskANN, or project-level acceptance
- rewrite `benchmark_results/hnsw_p3_002_final_verdict.json`
- change public API, FFI, or persistence behavior
- change the current round-6 or round-7 search-path evidence into verdict claims
- bundle batch-4 dispatch caching or AVX2 FMA specialization into the first graph-quality rerun

## Native Reference Surface

The reference insertion behavior remains the native HNSW add-point flow in:

- `/Users/ryan/Code/vectorDB/knowhere/thirdparty/hnswlib/hnswlib/hnswalg.h`

The reference behaviors to match are:

- greedy descent from `maxlevel` down to `curlevel + 1` before candidate search on lower layers
- overflow pruning through diversification heuristic rather than naive nearest-distance truncation

Round 8 does not require byte-for-byte structural identity with native. It only requires the Rust parallel build path to stop deviating on these two graph-quality-critical decisions.

## Rust Surface

The Rust surfaces in scope are:

- `find_neighbors_for_insertion_with_scratch()` in `src/faiss/hnsw.rs`
- `add_connections_for_node()` in `src/faiss/hnsw.rs`
- any small helper extracted to share descent or overflow-shrink logic across profiled and non-profiled bulk-build paths
- round-8 regression/profile tests and artifacts

The serial insertion helpers in `insert_node_with_scratch()` and the existing search-path micro-optimizations are reference points, not the primary round-8 target.

## Architecture

Round 8 keeps three layers separated:

- Historical truth layer: the archived HNSW, core-path, leadership, and production-acceptance verdict artifacts remain unchanged.
- Reopen evidence layer: add round-8-specific baseline, build audit, and authority summary artifacts.
- Algorithm layer: constrain code changes to the parallel build path and the upper-layer overflow shrink behavior used by that path.

The round-8 implementation should prefer extracting small helpers over scattering special cases:

- a helper that performs upper-layer greedy descent for insertion
- a helper that records or applies heuristic shrink for non-layer0 overflow during bulk build

## Round 8 Hypothesis

The next honest HNSW hypothesis is:

> Search-side structure is now close enough that the remaining recall-efficiency gap is dominated by graph quality, and the largest graph-quality drift in the current Rust implementation is the parallel build path's missing greedy upper-layer descent plus its naive upper-layer overflow truncation.

If this hypothesis is correct, then:

- round-8 build audit fields should show the parallel path now matches the intended descent/shrink modes
- low-`ef` search quality should improve on regression fixtures
- the remote same-schema lane should need less extra search work to maintain recall, producing a materially better Rust QPS row without recall regression

If this hypothesis is wrong, round 8 should prove that cleanly and leave the remaining work to a later, separately tracked search-side micro-optimization or larger memory-layout redesign.

## Feature Boundaries

### 1. Round 8 Activation

Freeze the current reopen baseline for this line and explicitly document that:

- round 5 stability did not produce attributable authority improvement
- round 6 and round 7 improved search-path evidence only
- the new target is `parallel_build_graph_quality_parity`
- historical family verdict remains `functional-but-not-leading`

This feature claims no algorithm win. It only reopens the queue around a new hypothesis.

### 2. Parallel-Build Graph Audit

Add a round-8 audit/profile surface that records:

- whether parallel insertion performs greedy upper-layer descent before searching `node_level`
- whether upper-layer overflow uses heuristic shrink or naive truncation
- build-time counters or metadata that show the new path is actually exercised
- reference links to the native insertion code and the serial Rust insertion path

This feature exists to make the graph-quality rework measurable and falsifiable.

### 3. Parallel-Build Graph Rework

Rework the Rust bulk-build path while preserving public contracts. Allowed changes include:

- adding greedy descent from `self.max_level` down to `node_level + 1` before the per-layer candidate search loop
- replacing upper-layer `truncate_to_best()` overflow handling with heuristic shrink behavior aligned with the serial path
- refactoring bulk-build bookkeeping so overflow shrink can be applied after connection insertion without borrow conflicts

Round 8 should avoid:

- changing search API behavior
- changing serialization format
- changing round-6/round-7 search-loop structure
- widening the scope to batch-4 dispatch caching or AVX2 FMA specialization

### 4. Authority Same-Schema Rerun

Re-run the real remote same-schema recall-gated lane and record whether graph-quality alignment materially improves Rust-side evidence. The round-8 authority summary should answer:

- did the Rust row move materially
- did recall stay at the gate
- is a later verdict-refresh feature justified
- should the next action be `continue`, `soft_stop`, or `hard_stop`

Round 8 itself does not rewrite the historical family or project verdict chain.

## Test Strategy

Round 8 uses three verification layers:

- default-lane round-8 contract tests that require the baseline, audit, and authority summary artifacts
- focused HNSW library regressions proving the parallel build path now performs upper-layer greedy descent and upper-layer heuristic overflow shrink
- remote x86 authority reruns on the same HDF5 and native capture surfaces already used by the earlier reopen rounds

The focused regressions should prefer deterministic small fixtures that compare:

- serial insert and parallel build graph shapes at upper layers
- search recall or candidate quality at modest `ef`
- overflow shrink behavior on synthetic upper-layer neighbor sets

## New Artifacts

Round 8 should add:

- `benchmark_results/hnsw_reopen_round8_baseline.json`
- `benchmark_results/hnsw_reopen_parallel_build_audit_round8.json`
- `benchmark_results/hnsw_reopen_round8_authority_summary.json`

The round-8 summary artifact must record:

- the new target
- the exact build-path mismatches being tested
- fresh same-schema Rust and native recall/QPS evidence
- deltas versus the new round-8 baseline
- whether verdict refresh is justified
- the next tracked action

## Stop/Go Rules

### Go

- the build audit confirms the parallel path now uses descent plus heuristic shrink
- focused regressions show graph-quality-sensitive behavior improves or stays aligned
- the remote same-schema lane materially improves while recall stays at the gate

### Soft Stop

- the graph-quality audit and focused tests improve, but the authority same-schema lane only moves modestly or remains difficult to attribute

### Hard Stop

- recall regresses materially
- the authority same-schema lane regresses again
- or the build-path alignment lands cleanly but still fails to move the trusted lane enough to justify this hypothesis

## Expected Outcome

At the end of round 8 the repository should be able to say one of three things honestly:

1. parallel-build graph-quality alignment materially improved the authority lane and merits a later verdict-refresh feature
2. the graph-quality fixes landed correctly but did not move authority evidence enough, so build parity is not the main remaining blocker
3. the hypothesis was wrong or insufficient, and the next tracked work must shift to a different bottleneck such as batch-4 dispatch caching, AVX2/FMA dispatch splitting, or a larger memory-layout cut
