# Final Production Acceptance Design

## Goal

Close `final-production-acceptance` by archiving the honest project-level verdict from the current remote x86 evidence: the production engineering gates are closed, but the project is not accepted because the leadership criterion is still unmet.

## Problem

All prerequisite work is now closed:

- core-path classifications are archived
- the final performance-leadership proof is archived as unmet
- the production governance gates are closed

What remains open is the project-level verdict. Leaving it as a generic `failing` feature would hide the actual conclusion. The repository needs a replayable final artifact that says clearly:

- the supporting production gates are closed
- the project is still not accepted on the current evidence
- no positive completion claim is allowed

## Options

### Option 1: Honest negative final verdict

Create a final production-acceptance artifact and a default-lane regression that archive `production_accepted=false`.

Pros:

- closes the last feature without inventing positive evidence
- keeps the current evidence chain replayable
- matches the already-archived `criterion_met=false` leadership proof

Cons:

- the project ends in a negative acceptance verdict rather than a celebratory completion

### Option 2: Leave the feature failing indefinitely

Pros:

- no extra artifact work

Cons:

- hides the actual conclusion
- leaves the queue permanently half-open even though the evidence is already decisive

### Option 3: Reopen performance work first

Pros:

- could theoretically recover a positive acceptance verdict later

Cons:

- contradicts the current evidence
- expands scope far beyond a final-acceptance closure

## Decision

Take Option 1.

This feature should close as a truthful project-level no-go for acceptance on the current remote evidence.

## Scope

This feature will:

1. add a default-lane regression that requires a final production-acceptance artifact
2. create `benchmark_results/final_production_acceptance.json`
3. verify the recorded remote commands:
   - `cargo fmt --all -- --check`
   - `cargo clippy --all-targets --all-features -- -D warnings`
   - `scripts/gate_profile_runner.sh --profile full_regression`
   - `python3 scripts/validate_features.py feature-list.json`
4. sync durable state so the project queue closes with an explicit `not accepted` verdict

## Non-Goals

This feature will not:

- reopen HNSW / IVF-PQ / DiskANN tuning
- invent a positive leadership claim
- rewrite the already-settled family artifacts

## Design

### 1. Final acceptance artifact

Add `benchmark_results/final_production_acceptance.json` with:

- `production_accepted=false`
- `acceptance_status=not_accepted`
- references to `final_core_path_classification` and `final_performance_leadership_proof`
- the list of production gates already closed
- explicit blocking reasons showing why the project still cannot claim acceptance

### 2. Regression tied to full regression

Add `tests/test_final_production_acceptance.rs` so `cargo test --tests -q` keeps the final artifact honest.

This regression should assert:

- the artifact exists and parses
- the verdict is `not_accepted`
- all production gates are marked closed
- replaceability and leadership requirements remain false

### 3. Final durable closure

Once the artifact and regression are green and the recorded remote commands pass:

- mark `final-production-acceptance` as `passing`
- mark the project queue closed with an explicit negative verdict
- move `task-progress.md` into an all-features-passing state rather than pointing at another queued feature

## Testing Strategy

1. add the new regression
2. run it red because the artifact is missing
3. add the artifact
4. run the regression green
5. replay the remote final-acceptance verification commands
6. sync durable state and rerun `scripts/validate_features.py`

## Risks

### Risk: confusing feature completion with project acceptance

Mitigation:

State this explicitly in the artifact and durable docs: the feature passes because the verdict is archived honestly, not because the project is accepted.

### Risk: final docs drift away from the underlying artifacts

Mitigation:

Point the new artifact directly at `final_core_path_classification.json` and `final_performance_leadership_proof.json`, and keep a test on the new artifact in `cargo test --tests`.
