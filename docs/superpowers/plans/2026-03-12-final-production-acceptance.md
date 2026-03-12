# Final Production Acceptance Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans or superpowers:subagent-driven-development before changing files from this plan.

**Goal:** Archive the final project-level verdict as `not accepted` on the current remote x86 evidence, then close the last remaining feature.

**Architecture:** Reuse the already-closed final-classification and leadership-proof artifacts as upstream truth. Add one small final artifact plus one regression test that gets exercised by the existing `full_regression` lane. Then replay the recorded remote commands and sync durable state.

**Tech Stack:** Rust integration tests, JSON benchmark artifacts, remote authority wrappers, durable long-task state files

---

## Chunk 1: Add The Final Regression

### Task 1: Require a final production-acceptance artifact

**Files:**
- Create: `tests/test_final_production_acceptance.rs`

- [ ] **Step 1: Add the failing regression**

Add a test that loads `benchmark_results/final_production_acceptance.json` and asserts:

- `production_accepted = false`
- the production gates are all marked `closed`
- the artifact references `final_core_path_classification.json`
- the artifact references `final_performance_leadership_proof.json`

- [ ] **Step 2: Run the regression RED**

Run: `cargo test --test test_final_production_acceptance -- --nocapture`
Expected: FAIL because the artifact does not exist yet.

## Chunk 2: Add The Final Artifact

### Task 2: Archive the project-level verdict

**Files:**
- Create: `benchmark_results/final_production_acceptance.json`

- [ ] **Step 1: Add the artifact**

Create `benchmark_results/final_production_acceptance.json` with:

- `production_accepted=false`
- `acceptance_status=not_accepted`
- references to the current final-classification and leadership-proof artifacts
- a closed list of production gates
- explicit blocking reasons for the negative verdict

- [ ] **Step 2: Run the regression GREEN**

Run: `cargo test --test test_final_production_acceptance -- --nocapture`
Expected: PASS

## Chunk 3: Replay The Recorded Authority Gate

### Task 3: Run the final recorded verification steps

**Files:**
- Inspect: `scripts/gate_profile_runner.sh`

- [ ] **Step 1: Refresh the remote workspace**

Run: `bash init.sh`
Expected: PASS

- [ ] **Step 2: Replay remote fmt**

Run: `bash scripts/remote/test.sh --command "cargo fmt --all -- --check"`
Expected: `test=ok`

- [ ] **Step 3: Replay remote clippy**

Run: `bash scripts/remote/test.sh --command "cargo clippy --all-targets --all-features -- -D warnings"`
Expected: `test=ok`

- [ ] **Step 4: Replay remote full regression**

Run: `bash scripts/remote/test.sh --command "scripts/gate_profile_runner.sh --profile full_regression"`
Expected: `test=ok`

- [ ] **Step 5: Validate durable consistency**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`

## Chunk 4: Close The Long Task Durably

### Task 4: Mark the last feature passing and archive the program verdict

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `README.md`

- [ ] **Step 1: Mark `final-production-acceptance` passing**

Close the feature because the final verdict is now archived.

- [ ] **Step 2: Record the project-level negative verdict**

Update durable docs so they explicitly say:

- all tracked features are closed
- the current program verdict is `not accepted`
- the blocker is the unmet leadership criterion already archived in the upstream final-proof artifact

- [ ] **Step 3: End with no queued feature**

Move `task-progress.md` and the queue/roadmap text into an all-features-closed state rather than pointing at another pending feature.
