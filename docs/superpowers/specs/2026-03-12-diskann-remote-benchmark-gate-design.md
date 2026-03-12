# DiskANN Remote Benchmark Gate Design

## Goal

Close `diskann-remote-benchmark-gate` by making the recorded benchmark entrypoints execute real default-lane regressions and by archiving explicit remote benchmark-gate evidence that DiskANN remains constrained and not native-comparable on the current authority surface.

## Problem

The current durable state says DiskANN already has an honest scope boundary:

- `src/faiss/diskann.rs` is still a simplified Vamana + placeholder PQ path
- `src/faiss/diskann_aisaq.rs` exposes a constrained AISAQ-like skeleton, not a native SSD DiskANN pipeline
- `tests/bench_compare.rs` already keeps DiskANN out of the native-comparable compare lane

But the specific benchmark-gate feature is not actually closed in `feature-list.json`, and its verification surface is still weaker than it should be:

- `tests/bench_diskann_1m.rs` currently guards scope wording, but does not lock a benchmark-gate conclusion
- `tests/bench_recall_gated_baseline.rs` and `tests/bench_cross_dataset_sampling.rs` currently lock IVF-PQ facts, not DiskANN benchmark facts
- `benchmark_results/cross_dataset_sampling.json` does not currently include DiskANN rows, so one third of the recorded verification surface is not yet about the DiskANN family at all

That leaves a workflow gap: the repo says DiskANN is constrained/no-go in several docs, but the benchmark-gate feature does not yet have a crisp, replayable artifact and default-lane checks to back that state.

## Decision

Use the benchmark gate to archive **explicit no-go benchmark evidence under constrained scope**, not the final family classification.

This keeps the feature boundaries clean:

- `diskann-remote-benchmark-gate` proves the benchmark gate outcome is explicit and replayable
- `diskann-stop-go-verdict` remains responsible for the family-level final classification (`constrained` or `no-go`)

## Scope

This feature will do four things:

1. Add a DiskANN benchmark-gate artifact that summarizes the current authority-backed benchmark evidence and constrained comparability boundary.
2. Extend the cross-dataset benchmark artifact to include DiskANN rows.
3. Upgrade the three recorded verification entrypoints into real default-lane regressions for DiskANN.
4. Sync durable governance docs so benchmark-gate closure is no longer left as an open question.

## Non-Goals

This feature will not:

- declare the final family-level DiskANN verdict
- add DiskANN back into the native-comparable compare lane
- claim SSD-pipeline parity or performance leadership
- rewrite the overall benchmark methodology
- broaden DiskANN implementation scope beyond the already-archived constrained boundary

## Approach

### 1. Add a benchmark-gate artifact

Create `benchmark_results/diskann_p3_004_benchmark_gate.json` with:

- `task_id = DISKANN-P3-004`
- `family = DiskANN`
- `benchmark_gate_verdict = no_go_for_native_comparable_benchmark`
- `comparability_status = constrained`
- references to the recall-gated baseline artifact, the cross-dataset artifact, and the constrained-scope regression surfaces
- a short summary explaining that DiskANN benchmark evidence is archived as explicit no-go under constrained scope

This artifact is intentionally not the final family verdict. It records benchmark-gate closure only.

### 2. Extend `cross_dataset_sampling` to include DiskANN

Update `src/benchmark/cross_dataset_sampling.rs` so each sampled dataset emits a DiskANN row built from the constrained `PQFlashIndex` path.

The important requirement is not that DiskANN performs well. The requirement is that the artifact now explicitly includes DiskANN and that the resulting rows remain below the recall gate or otherwise non-trusted, matching the current no-go benchmark story.

### 3. Turn the three verification entrypoints into real DiskANN regressions

The recorded verification files should each lock one piece of the benchmark-gate contract:

- `tests/bench_diskann_1m.rs`: benchmark-gate artifact exists and says DiskANN is constrained / no-go for native-comparable benchmarking
- `tests/bench_recall_gated_baseline.rs`: baseline artifact still includes a DiskANN row that remains sub-gate and non-trusted
- `tests/bench_cross_dataset_sampling.rs`: cross-dataset artifact now includes DiskANN rows on all sampled datasets, and each row remains sub-gate or non-trusted

Heavy benchmark generators stay behind `feature = "long-tests"` and `#[ignore]`.

### 4. Refresh the cross-dataset artifact on the authority host

Because the new DiskANN cross-dataset rows are benchmark evidence, the artifact refresh should be produced on the remote x86 authority host and then copied back into the repo.

This is the only part of the feature that needs a heavy benchmark rerun. The rest of the verification surface remains the fast default-lane regression set already recorded in `feature-list.json`.

## Testing Strategy

Use TDD:

1. Add failing default-lane DiskANN regressions first.
2. Add the benchmark-gate artifact and DiskANN cross-dataset support.
3. Regenerate the cross-dataset artifact on the authority host and sync it back.
4. Run the local fast regressions.
5. Run the recorded remote verification commands.

## Risks

### Risk: Collapsing benchmark gate and final family verdict into one feature

Mitigation:

Use `benchmark_gate_verdict` / `comparability_status` terminology in the new artifact and leave final classification to `diskann-stop-go-verdict`.

### Risk: Local-only benchmark artifact drift

Mitigation:

Refresh the changed cross-dataset artifact on the remote authority host and cite that run in durable state.

### Risk: Weak verification despite new docs

Mitigation:

Every recorded verification entrypoint must execute a real default-lane regression tied to DiskANN benchmark facts, not merely compile a file or preserve a wording note.
