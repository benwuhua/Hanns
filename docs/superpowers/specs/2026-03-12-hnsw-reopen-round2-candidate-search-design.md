# HNSW Reopen Round 2 Candidate-Search Design

## Goal

Reopen HNSW for a second focused algorithm line whose primary success criterion is improved Rust QPS on the remote x86 same-schema recall-gated authority lane, without recall regression or API/FFI/persistence contract breakage.

## Context

The first HNSW reopen line closed honestly but did not move the family verdict. The authority refresh recorded in `benchmark_results/hnsw_reopen_profile_round1.json` showed:

- profiled build wall clock improved from `17006.687ms` to `16018.719ms`
- `neighbor_selection`, `connection_update`, and `layer_descent` all shrank
- `candidate_search` grew from `8667.369ms` to `9086.895ms`
- sample-search qps fell from `1410.104` to `1329.255`

The historical HNSW family verdict in `benchmark_results/hnsw_p3_002_final_verdict.json` therefore remains `functional-but-not-leading`, with the remote same-schema evidence still showing native about `14.8x` faster at the trusted recall gate.

## Decision

Round 2 will optimize both build and search paths, but it will judge success primarily by the remote same-schema recall-gated QPS result. Build-only improvements are useful only if they support that final authority result.

## Scope

Round 2 includes four tracked features:

1. `hnsw-reopen-round2-activation`
2. `hnsw-candidate-search-profiler`
3. `hnsw-candidate-search-core-rework`
4. `hnsw-round2-authority-same-schema-rerun`

Round 2 does not reopen IVF-PQ, DiskANN, or the project-level final acceptance chain. Existing final artifacts remain historical truth unless new authority evidence justifies rewriting them.

## Architecture

Round 2 keeps three layers separate:

- Historical truth layer: keep `benchmark_results/hnsw_p3_002_final_verdict.json`, `benchmark_results/final_core_path_classification.json`, and `benchmark_results/final_performance_leadership_proof.json` unchanged unless fresh authority evidence warrants a rewrite.
- Reopen tracking layer: add round-2-specific artifacts that describe the candidate-search hypothesis, the profiling breakdown, and the authority rerun conclusion.
- Algorithm layer: concentrate implementation changes inside `src/faiss/hnsw.rs`, specifically in the shared candidate-search core used by both build-time insertion and query-time traversal.

## Round 2 Hypothesis

The next material QPS gain is most likely behind the shared candidate-search core rather than behind another build-only graph-maintenance tweak. The current evidence suggests the first reopen line improved some secondary costs but left the dominant `candidate_search` cost unresolved, and may have made it worse.

## Feature Boundaries

### 1. Round 2 Activation

Freeze the end of round 1 into a new round-2 baseline artifact and durable workflow state. This feature does not claim performance improvement; it only changes the active hypothesis from "first reopen build-path rework" to "candidate-search-centered round 2."

### 2. Candidate-Search Profiler

Add a deeper profiling surface that splits candidate search into smaller buckets such as:

- entry-point descent and level hopping
- frontier push/pop activity
- visited-state marking/reset
- distance computation
- pruning/result-heap maintenance

This feature exists to make the next algorithm cut measurable and falsifiable.

### 3. Candidate-Search Core Rework

Rework the shared candidate-search core in `src/faiss/hnsw.rs` with minimal semantic drift. Allowed targets include:

- scratch lifetime and reuse
- visited-state reset strategy
- candidate/result heap behavior
- shared candidate-expansion logic used by build and search

Round 2 should avoid mixing in unrelated graph-topology rewrites unless the profiler demonstrates they are part of the same bottleneck.

### 4. Authority Same-Schema Rerun

Re-run the real recall-gated same-schema remote lane and summarize whether round 2 produced enough improvement to revisit the HNSW family verdict. Synthetic profiling alone is insufficient.

## Test Strategy

Round 2 uses three verification layers:

- default-lane artifact/contract tests
- focused HNSW unit and regression tests around candidate-search semantics
- remote x86 authority replays, with same-schema compare as the acceptance surface

The historical compare tests stay in place so the repository cannot silently rewrite the previous HNSW conclusion without new evidence.

## New Artifacts

Round 2 should add:

- `benchmark_results/hnsw_reopen_round2_baseline.json`
- `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json`
- `benchmark_results/hnsw_reopen_round2_authority_summary.json`

The final summary artifact must record:

- the round-2 target
- the latest remote same-schema recall/QPS result
- deltas versus the reopen baseline
- whether the HNSW family verdict may be rewritten
- whether the next step is continue, soft stop, or hard stop

## Stop/Go Rules

### Go

- remote same-schema recall gate holds and Rust QPS improves materially
- or the new candidate-search profile shows a clear reduction in the dominant hotspot without introducing a larger replacement hotspot

### Soft Stop

- local/profile evidence improves but same-schema authority QPS does not move enough
- in this case only one more round is justified, and it must use a new explicit hypothesis rather than more vague candidate-search tuning

### Hard Stop

- recall gate regresses materially
- or two consecutive rounds against the same core hypothesis fail to improve the authority lane
- or progress requires breaking the current API/FFI/persistence contract

## Expected Outcome

At the end of round 2 the repository should be able to say one of three things honestly:

1. candidate-search rework materially improved authority performance and HNSW may be reclassified
2. round 2 improved local/profiling signals but not enough to move the authority verdict
3. the candidate-search hypothesis did not pay off and HNSW should stop or change hypotheses
