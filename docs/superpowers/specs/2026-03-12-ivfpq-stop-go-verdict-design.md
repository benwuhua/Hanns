# IVF-PQ Stop-Go Verdict Design

## Goal

Close the IVF-PQ family with an honest, replayable final classification backed by existing remote-authority artifacts, and eliminate the remaining false-green benchmark verification shells in the default regression lane.

## Problem

The repository already contains enough IVF-PQ evidence to support a family verdict:

- `benchmark_results/ivfpq_p1_002_focused.json` reports recall@10 around `0.442`
- `benchmark_results/recall_gated_baseline.json` reports IVF-PQ recall@10 around `0.47`, below the `0.8` recall gate
- `benchmark_results/cross_dataset_sampling.json` shows IVF-PQ failing to reach the recall gate consistently across multiple dataset shapes
- the production-facing contract feature `ivfpq-ffi-persistence-contract` is already closed

But the current `ivfpq-stop-go-verdict` verification entrypoints still point at test files that compile away on the default lane because they are gated by file-level `#![cfg(feature = "long-tests")]`. That means the durable workflow still has false-green surface area for the IVF-PQ family verdict.

## Decision

Archive IVF-PQ as `no-go`.

This is stricter than `functional-but-not-leading` and more honest than `constrained` for the current evidence set. The problem is not only that IVF-PQ trails native or lacks proof of leadership; it is that the current trusted artifacts do not satisfy the minimum recall gate required for credible production or benchmark claims.

## Why `no-go`

The existing evidence is consistent:

- baseline artifact: recall below gate with `confidence = recheck required`
- focused hot-path artifact: recall still below gate on the real `IvfPqIndex` path
- cross-dataset artifact: no stable dataset shape where IVF-PQ clears the gate with trusted confidence

Given that consistency, this feature should not reopen heavy benchmark work. It should formalize the current end-state and lock it with default-lane regressions.

## Non-Goals

This feature will not:

- improve IVF-PQ recall or throughput
- rerun the heavy remote benchmark generators unless a verification step explicitly requires existing artifact-based checks
- change the underlying benchmark methodology
- relitigate whether `src/faiss/ivf.rs` is the real hot path

## Approach

### 1. Add a final IVF-PQ family verdict artifact

Create `benchmark_results/ivfpq_p3_003_final_verdict.json` with:

- `task_id = IVFPQ-P3-003`
- `family = IVF-PQ`
- `classification = no-go`
- `leadership_verdict = no_go_for_production_and_leadership`
- `leadership_claim_allowed = false`
- references to the existing authoritative benchmark artifacts and contract evidence
- a short summary and action statement explaining that the family is archived rather than promoted

### 2. Convert benchmark tests into default-lane regressions

Keep the heavy generators behind `feature = "long-tests"` and `#[ignore]`, but add default-lane checks that:

- verify the new family verdict artifact is present and structurally coherent
- verify the baseline artifact still records IVF-PQ below the recall gate with non-trusted confidence
- verify the cross-dataset artifact still records IVF-PQ as sub-gate or non-trusted across all sampled datasets

This mirrors the HNSW family verdict closure pattern and removes another class of `0 tests` false green.

### 3. Sync durable governance state

Update the durable project files so they all tell the same story:

- `feature-list.json`
- `task-progress.md`
- `RELEASE_NOTES.md`
- `TASK_QUEUE.md`
- `GAP_ANALYSIS.md`
- `docs/PARITY_AUDIT.md`

The queue/gap/audit layer should explicitly state that IVF-PQ is archived as `no-go`, not left as an open benchmark question.

## Testing Strategy

Use TDD:

1. Add failing default-lane regressions first.
2. Add the verdict artifact and any tiny helper logic needed for the tests.
3. Run the focused local tests until green.
4. Run the feature verification steps on the remote authority host exactly as recorded in `feature-list.json`.

## Risks

### Risk: Overstating certainty

Mitigation:

The artifact should say `no-go` based on the current authority evidence set, not claim impossibility of future improvement.

### Risk: Reintroducing false-green benchmark lanes

Mitigation:

Make every recorded verification entrypoint execute a real default-lane regression, not a file-level `long-tests` shell.

### Risk: Durable docs drifting again

Mitigation:

Update queue/gap/parity state in the same change and re-run `python3 scripts/validate_features.py feature-list.json` before handoff.
