# HNSW Reopen Round 3 Distance-Compute Design

## Goal

Reopen HNSW for a third focused algorithm line whose primary success criterion is improved Rust QPS on the remote x86 same-schema recall-gated authority lane, without recall regression and without changing the current API, FFI, persistence, or family/project verdict artifacts.

## Context

Round 2 closed honestly as a `hard_stop`. The authority summary in `benchmark_results/hnsw_reopen_round2_authority_summary.json` recorded:

- synthetic candidate-search profiling improved materially
- the authoritative Rust same-schema HNSW row regressed from `710.962` to `521.031` qps
- recall moved only slightly upward from `0.9915` to `0.9923`
- native BF16 remained essentially flat at `10519.683` qps
- the native-over-Rust throughput gap therefore widened from `14.8x` to `20.2x`

The refreshed round-2 profile in `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json` also changed the hotspot ranking. `entry_descent` is no longer the dominant cost. The largest remaining measured bucket is now `distance_compute` at about `39.1ms` and about `58.8%` of profiled time. `visited_ops` and `frontier_ops` are secondary costs.

The historical HNSW family verdict in `benchmark_results/hnsw_p3_002_final_verdict.json` therefore remains `functional-but-not-leading`, and round 3 must begin from a new, narrower hypothesis rather than implicitly continuing round 2.

## Decision

Round 3 will target `distance_compute_inner_loop` specifically. It will not attempt another broad search-path rewrite. The round-3 line exists to answer a narrower question: can a focused L2 distance-compute fast path and related hot-loop specialization improve the authoritative same-schema HNSW lane without semantic drift?

## Scope

Round 3 includes four tracked features:

1. `hnsw-reopen-round3-activation`
2. `hnsw-distance-compute-profiler`
3. `hnsw-distance-l2-fast-path-rework`
4. `hnsw-round3-authority-same-schema-rerun`

Round 3 does not reopen IVF-PQ, DiskANN, the project-level final acceptance chain, or the HNSW family verdict itself. Existing final artifacts remain historical truth unless a later tracked feature explicitly rewrites them using fresh authority evidence.

## Architecture

Round 3 keeps three layers separate:

- Historical truth layer: `benchmark_results/hnsw_p3_002_final_verdict.json`, `benchmark_results/final_core_path_classification.json`, `benchmark_results/final_performance_leadership_proof.json`, and `benchmark_results/final_production_acceptance.json` remain unchanged throughout round 3.
- Reopen tracking layer: add round-3-specific artifacts that freeze the new baseline, profile `distance_compute` more precisely, and summarize the same-schema authority rerun.
- Algorithm layer: keep implementation changes inside `src/faiss/hnsw.rs`, specifically in the query-time candidate-search hot loop and the helper functions that feed it.

## Round 3 Hypothesis

The next honest HNSW performance hypothesis is no longer "candidate search in general" but "the current L2 distance inner loop and its immediate call sites are too expensive." The profile evidence suggests that repeated `query -> node` distance evaluation in upper-layer descent and layer-0 candidate expansion is now the largest measurable cost center. If that hypothesis is wrong, round 3 should prove it quickly and stop.

## Feature Boundaries

### 1. Round 3 Activation

Freeze round 2 as the historical baseline for this line and add a new round-3 activation artifact that records:

- round 2 ended in `hard_stop`
- the new target is `distance_compute_inner_loop`
- the historical HNSW verdict remains unchanged

This feature does not claim improvement. It only changes the active hypothesis and durable workflow state.

### 2. Distance-Compute Profiler

Add a deeper profiling surface that splits the current monolithic `distance_compute` bucket into smaller sources such as:

- upper-layer greedy `query -> node` distance calls
- layer-0 candidate-expansion `query -> node` distance calls
- `node -> node` distance calls used for graph maintenance or helper logic
- call counts for each sub-path

This feature exists to make the next algorithm cut measurable and falsifiable.

### 3. Distance L2 Fast-Path Rework

Rework the hottest L2 search path in `src/faiss/hnsw.rs` while preserving results and external contracts. Allowed targets include:

- one-time outer dispatch for the common `L2 + no filter` path
- specialized internal helpers for L2 `query -> node` distance computation
- tighter hot-loop structure in upper-layer greedy descent and layer-0 candidate expansion
- small adjacent reductions in `visited_ops` or `frontier_ops` only when they are part of the same hot loop

Round 3 should avoid:

- changing graph construction semantics
- changing vector storage format on disk
- changing the public API or FFI contract
- mixing in broad layout rewrites or unrelated build-path changes

### 4. Authority Same-Schema Rerun

Re-run the real recall-gated same-schema remote lane and summarize whether round 3 produced enough improvement to justify opening a later verdict-refresh feature. Round 3 itself does not rewrite the family or project verdict chain.

## Test Strategy

Round 3 uses three verification layers:

- default-lane artifact/contract tests for round-3 activation, profile, and authority summary
- focused HNSW unit/regression tests proving the L2 fast path preserves search behavior
- remote x86 authority replays, with the same-schema HDF5 compare surface as the acceptance lane

Historical compare tests remain in place so the repository cannot silently rewrite the current HNSW conclusion without new authority evidence.

## New Artifacts

Round 3 should add:

- `benchmark_results/hnsw_reopen_round3_baseline.json`
- `benchmark_results/hnsw_reopen_distance_compute_profile_round3.json`
- `benchmark_results/hnsw_reopen_round3_authority_summary.json`

The final round-3 summary artifact must record:

- the round-3 target
- fresh same-schema Rust and native recall/QPS evidence
- deltas versus the round-2 hard-stop baseline
- whether a later verdict-refresh feature is justified
- whether the next action is `continue`, `soft_stop`, or `hard_stop`

## Stop/Go Rules

### Go

- the round-3 synthetic profile shows that `distance_compute` or its dominant sub-paths got materially cheaper, and
- the remote same-schema recall gate holds while Rust QPS improves materially relative to round 2

### Soft Stop

- synthetic/profile evidence improves clearly, but the authority same-schema lane barely moves
- in this case only one new explicit hypothesis is justified; round 3 itself does not auto-continue

### Hard Stop

- recall regresses materially
- or the authority same-schema lane regresses again
- or the new profiler shows `distance_compute` was not the real limiting factor
- or further progress would require changing the current API, FFI, persistence, or file format contract

## Expected Outcome

At the end of round 3 the repository should be able to say one of three things honestly:

1. the distance-compute hypothesis materially improved the authority lane and merits a later verdict-refresh feature
2. the distance-compute hypothesis improved synthetic or local signals but not enough to move the authority result
3. the distance-compute hypothesis did not pay off and HNSW should stop again or change hypotheses
