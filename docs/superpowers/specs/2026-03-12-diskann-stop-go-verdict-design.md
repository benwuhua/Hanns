# DiskANN Stop-Go Verdict Design

## Goal

Close `diskann-stop-go-verdict` by archiving a family-level remote-only verdict for DiskANN that is consistent with the already-closed benchmark gate, scope audits, and compare-lane constraints.

## Problem

The repo now has enough authority-backed evidence to say two things clearly:

- DiskANN is not a native-comparable benchmark candidate on the current Rust path.
- DiskANN still exposes real, testable functionality through a simplified Vamana path (`DiskAnnIndex`) and a constrained AISAQ skeleton (`PQFlashIndex`).

That leaves a final classification gap. `diskann-remote-benchmark-gate` archived the benchmark lane as explicit no-go evidence under constrained scope, but it intentionally did not decide the family-level classification. The remaining feature must answer the narrower question: is the DiskANN family itself `constrained` or `no-go`?

## Decision

Classify the DiskANN family as **`constrained`**, not `no-go`.

The reasoning is:

- the benchmark lane is already a clear no-go for native-comparable claims
- both implementation paths remain simplified and non-native-comparable
- but the family still has real lifecycle/search/persistence surfaces and executable boundary audits, so `no-go` would overstate the evidence

This mirrors the structure already used elsewhere in the repo:

- HNSW: family usable, leadership claim blocked
- IVF-PQ: family no-go because the actual hot path still fails the recall gate
- DiskANN: family constrained because the active implementations are functional but fundamentally simplified

## Scope

This feature will:

1. add a family-level final verdict artifact for DiskANN
2. upgrade the existing verification entrypoints into real verdict regressions
3. sync durable governance docs so DiskANN is no longer described only through benchmark-gate language

## Non-Goals

This feature will not:

- reopen native-comparable DiskANN benchmark work
- move DiskANN into the compare lane
- broaden `src/faiss/diskann.rs` or `src/faiss/diskann_aisaq.rs` toward native parity
- rerun heavy benchmark generation beyond the already-refreshed authority artifacts

## Artifact Design

Add `benchmark_results/diskann_p3_004_final_verdict.json` with:

- `task_id = DISKANN-P3-004`
- `family = DiskANN`
- `classification = constrained`
- `leadership_verdict = no_go_for_native_comparable_benchmark`
- `leadership_claim_allowed = false`
- references to:
  - `benchmark_results/diskann_p3_004_benchmark_gate.json`
  - `benchmark_results/recall_gated_baseline.json`
  - `benchmark_results/cross_dataset_sampling.json`
  - the scope/contract verification surfaces

The summary should explicitly separate the two ideas:

- benchmark lane: no-go
- family lane: constrained, simplified, and not production-candidate for native parity claims

## Verification Design

The recorded feature verification already tells us where the final verdict must be locked:

### 1. `cargo test --lib diskann -- --nocapture`

Add or tighten library-level verdict coverage so the family verdict is grounded in executable scope facts:

- `DiskAnnIndex::scope_audit()` stays non-native-comparable and placeholder-PQ based
- `PQFlashIndex::scope_audit()` stays non-native-comparable even after mmap/page-cache persistence
- a focused regression reads the final verdict artifact and asserts it matches the library-level boundary

### 2. `cargo test --test bench_diskann_1m -q`

Upgrade the default-lane benchmark regression so it now locks both:

- the benchmark-gate artifact
- the family final verdict artifact

This keeps benchmark scope and family classification from drifting apart.

### 3. `cargo test --test bench_compare -q`

Add a final-verdict regression that requires:

- `classification = constrained`
- `leadership_claim_allowed = false`
- compare lane still excludes DiskANN

This turns the compare lane into a real policy lock rather than only a local helper assertion.

## Testing Strategy

Use TDD:

1. add failing final-verdict regressions to the three recorded verification surfaces
2. create the final verdict artifact and the minimum test helpers needed to satisfy those regressions
3. run the recorded local checks
4. run the recorded remote checks on authority
5. update durable workflow state

## Risks

### Risk: collapsing `constrained` into `no-go`

Mitigation:

Keep the artifact language explicit: native-comparable benchmarking is blocked, but the family still has functional constrained implementations.

### Risk: artifact-only verdict with no executable contract

Mitigation:

Bind the artifact to all three recorded verification surfaces, including `cargo test --lib diskann`.

### Risk: reopening benchmark scope by implication

Mitigation:

Require the final verdict summary and compare-lane regressions to keep `leadership_claim_allowed = false` and DiskANN excluded from native-comparable compare coverage.
