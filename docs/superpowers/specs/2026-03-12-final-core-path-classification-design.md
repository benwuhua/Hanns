# Final Core Path Classification Design

## Goal

Close `final-core-path-classification` by archiving a single remote-only summary of the final HNSW, IVF-PQ, and DiskANN family classifications, and binding that summary to the two benchmark artifacts already recorded in the feature verification surface.

## Problem

The repo now has honest family-level verdict artifacts for all three core CPU paths:

- HNSW: `functional-but-not-leading`
- IVF-PQ: `no-go`
- DiskANN: `constrained`

But those conclusions still live in separate family artifacts. The final-acceptance layer has no single artifact that says “these are the settled classifications of the core CPU paths,” and the recorded verification steps for this feature still only prove the baseline and cross-dataset artifacts exist, not that they align with the settled family verdicts.

That leaves a final governance gap: the family verdicts are individually closed, but the project still lacks a replayable cross-family classification input for final acceptance.

## Decision

Add one cross-family summary artifact and use the existing baseline and cross-dataset regressions to lock it.

This is intentionally narrow:

- no new heavy benchmark generation
- no change to existing family-level verdict artifacts
- no validator expansion
- just one explicit final-acceptance input that summarizes the three closed family verdicts and proves they still match the current authority-backed benchmark facts

## Scope

This feature will:

1. create a cross-family classification artifact in `benchmark_results/`
2. extend `tests/bench_recall_gated_baseline.rs` to lock that artifact against baseline facts
3. extend `tests/bench_cross_dataset_sampling.rs` to lock that artifact against cross-dataset facts
4. sync durable workflow state so the next final-acceptance feature can consume a single classification source

## Non-Goals

This feature will not:

- reopen HNSW, IVF-PQ, or DiskANN family verdicts
- rerun long benchmarks or refresh benchmark artifacts
- prove performance leadership
- decide final project acceptance

## Artifact Design

Add `benchmark_results/final_core_path_classification.json` with:

- `task_id = FINAL-CORE-CLASSIFICATION`
- `authority_scope = remote_x86_only`
- references to:
  - `benchmark_results/recall_gated_baseline.json`
  - `benchmark_results/cross_dataset_sampling.json`
  - `benchmark_results/hnsw_p3_002_final_verdict.json`
  - `benchmark_results/ivfpq_p3_003_final_verdict.json`
  - `benchmark_results/diskann_p3_004_final_verdict.json`
- one row per family with:
  - `family`
  - `classification`
  - `leadership_claim_allowed`
  - small evidence summary fields copied from the family artifacts

The artifact should explicitly summarize:

- HNSW is `functional-but-not-leading`
- IVF-PQ is `no-go`
- DiskANN is `constrained`

## Verification Design

The feature’s recorded verification steps already point at the right surfaces:

### 1. `python3 scripts/validate_features.py feature-list.json`

This continues to enforce durable-state consistency after the classification artifact and docs land.

### 2. `cargo test --test bench_recall_gated_baseline -q`

Upgrade the default-lane baseline regression so it also loads `final_core_path_classification.json` and asserts:

- HNSW classification is `functional-but-not-leading` while the baseline row remains trusted and above the recall gate
- IVF-PQ classification is `no-go` while the baseline row remains below the gate / non-trusted
- DiskANN classification is `constrained` while the baseline row remains below the gate / non-trusted

### 3. `cargo test --test bench_cross_dataset_sampling -q`

Upgrade the default-lane cross-dataset regression so it also asserts:

- HNSW summary remains `functional-but-not-leading` and all sampled rows remain trusted above the recall gate
- IVF-PQ summary remains `no-go` and sampled rows remain sub-gate or non-trusted
- DiskANN summary remains `constrained` and sampled rows remain sub-gate or non-trusted

## Testing Strategy

Use TDD:

1. add failing summary-artifact regressions to the two existing benchmark-lane tests
2. create the cross-family classification artifact
3. rerun local focused tests
4. rerun the recorded remote verification commands
5. update durable workflow state

## Risks

### Risk: summary artifact drifts from family artifacts

Mitigation:

Each regression will load the summary artifact and compare its classification fields against live benchmark facts that are already tied to the family verdicts.

### Risk: this becomes a hidden project-acceptance verdict

Mitigation:

Keep the artifact scoped to “core path classifications only.” Leave final project acceptance and performance leadership proof to later features.

### Risk: overfitting the summary to current benchmark details

Mitigation:

Lock only the facts needed to justify the classification categories, not every numeric field in every family artifact.
