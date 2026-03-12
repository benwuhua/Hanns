# Final Performance Leadership Proof Design

## Goal

Close `final-performance-leadership-proof` by archiving a single remote-only final-acceptance artifact that states the project does not currently satisfy the completion criterion of proving at least one core CPU path leads native on the same authority benchmark lane.

## Problem

The repo now has settled remote-only family outcomes for every core CPU path:

- HNSW: `functional-but-not-leading`
- IVF-PQ: `no-go`
- DiskANN: `constrained`

It also has a cross-family rollup artifact in `benchmark_results/final_core_path_classification.json`.

But the final-acceptance layer still lacks one explicit artifact that answers the remaining program-level question: "did the project prove at least one performance-leadership lane on the remote x86 authority surface?"

Without that explicit artifact, the completion criterion is only implied by separate family verdicts and the HNSW compare lane. The current verification steps for this feature also still point at HNSW evidence surfaces without a final artifact that ties them together into a replayable acceptance decision.

## Decision

Add one narrow final-proof artifact and bind it to the existing HNSW compare lane.

This feature will not reopen benchmarks or attempt new tuning. It will instead archive the current truth:

- HNSW remains below native throughput leadership on the trusted same-schema HDF5 lane
- IVF-PQ has already closed as `no-go`
- DiskANN has already closed as `constrained` and is not eligible for native-comparable leadership claims
- therefore `criterion_met = false` for the program's final performance-leadership completion criterion

## Scope

This feature will:

1. create a final acceptance artifact in `benchmark_results/`
2. extend `tests/bench_hnsw_cpp_compare.rs` so the default lane locks that artifact against the current HNSW baseline and cross-family rollup evidence
3. refresh the existing authority verification chain already recorded in `feature-list.json`
4. sync durable workflow state so later final-acceptance features consume an explicit completion-criterion outcome rather than inferring it

## Non-Goals

This feature will not:

- rerun or redesign the heavy HNSW benchmark methodology
- reopen HNSW, IVF-PQ, or DiskANN family verdicts
- claim the project is accepted overall
- create a synthetic "maybe" state for leadership proof

## Artifact Design

Add `benchmark_results/final_performance_leadership_proof.json` with:

- `task_id = FINAL-PERFORMANCE-LEADERSHIP-PROOF`
- `authority_scope = remote_x86_only`
- `criterion = at_least_one_core_cpu_path_beats_native_on_same_remote_benchmark_surface_at_same_recall_gate`
- `criterion_met = false`
- references to:
  - `benchmark_results/baseline_p3_001_stop_go_verdict.json`
  - `benchmark_results/hnsw_p3_002_final_verdict.json`
  - `benchmark_results/final_core_path_classification.json`
- one family entry each for HNSW, IVF-PQ, and DiskANN with:
  - `family`
  - `classification`
  - `leadership_claim_allowed`
  - `leadership_status`
  - concise blocker text

The artifact should make the final logic explicit:

- HNSW is the only core path with trusted same-schema compare evidence, and that evidence is still `no_go_for_performance_leadership`
- IVF-PQ is `no-go`, so it cannot satisfy the criterion
- DiskANN is `constrained`, and its benchmark lane is already closed as non-comparable / non-trusted for leadership

## Verification Design

Keep the feature's existing verification steps and make them enforce the final-proof artifact.

### 1. `bash scripts/remote/native_hnsw_qps_capture.sh --gtest-filter Benchmark_float_qps.TEST_HNSW`

This keeps the native HNSW authority lane fresh and preserves the upstream evidence source for the HNSW leadership blocker.

### 2. `bash scripts/remote/test.sh --command "cargo build --features hdf5 --bin generate_hdf5_hnsw_baseline --verbose"`

This keeps the Rust side of the same-schema HDF5 evidence chain buildable on the authority machine.

### 3. `bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"`

Upgrade the compare-lane regression so it also loads `final_performance_leadership_proof.json` and asserts:

- `criterion_met` is `false`
- the artifact points at the current HNSW stop-go and cross-family rollup sources
- HNSW remains blocked by the trusted native-over-Rust throughput gap
- IVF-PQ and DiskANN remain in non-leadership states consistent with the final core path rollup

## Testing Strategy

Use TDD:

1. add a failing final-proof regression to `tests/bench_hnsw_cpp_compare.rs`
2. run the compare lane to confirm the missing artifact fails for the expected reason
3. create `benchmark_results/final_performance_leadership_proof.json`
4. rerun the compare lane until it passes
5. rerun the recorded remote commands
6. update durable workflow state

## Risks

### Risk: the artifact duplicates family verdicts without adding signal

Mitigation:

Keep it narrowly scoped to the single project-level completion criterion that none of the existing family artifacts answer directly.

### Risk: the final-proof artifact drifts from the current HNSW authority evidence

Mitigation:

Make the compare-lane regression assert both the HNSW blocker source and the cross-family rollup source so the final-proof artifact cannot silently drift from the closed evidence chain.

### Risk: the artifact reads like overall project acceptance

Mitigation:

State only that the performance-leadership completion criterion is not met. Leave broader final-acceptance conclusions to later features.
