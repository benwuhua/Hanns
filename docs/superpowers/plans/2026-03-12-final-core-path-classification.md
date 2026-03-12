# Final Core Path Classification Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive a single remote-only summary of the final HNSW, IVF-PQ, and DiskANN core CPU path classifications and bind it to the existing baseline and cross-dataset verification surfaces.

**Architecture:** Keep the work narrow: create one classification artifact in `benchmark_results/`, then extend the two existing benchmark-lane regression files so they assert the summary remains aligned with the current authority-backed baseline and cross-dataset facts. Do not regenerate benchmark artifacts or reopen family verdicts.

**Tech Stack:** Rust 2021 integration tests in `tests/`, JSON verdict artifacts in `benchmark_results/`, durable workflow docs in repo root and `docs/`, remote authority wrappers in `scripts/remote/`

---

## Chunk 1: Add Failing Classification Summary Regressions

### Task 1: Make the recorded benchmark-lane verifications fail for the missing cross-family artifact

**Files:**
- Modify: `tests/bench_recall_gated_baseline.rs`
- Modify: `tests/bench_cross_dataset_sampling.rs`

- [ ] **Step 1: Add a failing baseline-summary regression**

Extend `tests/bench_recall_gated_baseline.rs` with a default-lane test that loads `benchmark_results/final_core_path_classification.json` and asserts:

- HNSW classification is `functional-but-not-leading`
- IVF-PQ classification is `no-go`
- DiskANN classification is `constrained`

and that those classifications still match the current baseline artifact rows.

- [ ] **Step 2: Add a failing cross-dataset-summary regression**

Extend `tests/bench_cross_dataset_sampling.rs` with a default-lane test that loads the same summary artifact and asserts the three family classifications remain aligned with the current sampled-row evidence.

- [ ] **Step 3: Run focused tests to verify RED**

Run: `cargo test --test bench_recall_gated_baseline -- --nocapture`
Expected: FAIL because `benchmark_results/final_core_path_classification.json` does not exist yet.

Run: `cargo test --test bench_cross_dataset_sampling -- --nocapture`
Expected: FAIL because `benchmark_results/final_core_path_classification.json` does not exist yet.

## Chunk 2: Add The Summary Artifact And Make The Regressions Pass

### Task 2: Implement the minimum cross-family classification closure

**Files:**
- Create: `benchmark_results/final_core_path_classification.json`
- Modify: `tests/bench_recall_gated_baseline.rs`
- Modify: `tests/bench_cross_dataset_sampling.rs`

- [ ] **Step 1: Create the cross-family classification artifact**

Add `benchmark_results/final_core_path_classification.json` with:

- `task_id = FINAL-CORE-CLASSIFICATION`
- references to the baseline, cross-dataset, and three family verdict artifacts
- one row each for HNSW, IVF-PQ, and DiskANN with their settled classifications and leadership-claim posture

- [ ] **Step 2: Keep the two summary regressions narrow**

Each new regression should only assert the facts needed to justify the current family classifications from the current benchmark artifacts, not every field from every source artifact.

- [ ] **Step 3: Run focused tests to verify GREEN**

Run: `cargo test --test bench_recall_gated_baseline -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_cross_dataset_sampling -- --nocapture`
Expected: PASS

## Chunk 3: Verify On Authority And Persist Durable State

### Task 3: Close the feature honestly in the long-task workflow

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `DEV_ROADMAP.md`

- [ ] **Step 1: Run the recorded local verification commands**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: PASS after durable-state updates

Run: `cargo test --test bench_recall_gated_baseline -q`
Expected: PASS

Run: `cargo test --test bench_cross_dataset_sampling -q`
Expected: PASS

- [ ] **Step 2: Refresh remote authority sync and run the recorded remote verification commands**

Run: `bash init.sh`
Expected: PASS

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_recall_gated_baseline -q"`
Expected: `test=ok`

Run: `bash scripts/remote/test.sh --command "cargo test --test bench_cross_dataset_sampling -q"`
Expected: `test=ok`

- [ ] **Step 3: Update durable workflow state**

Mark `final-core-path-classification` as `passing`, record the new cross-family classification artifact and authority logs in durable docs, set the next feature handoff, and re-run `python3 scripts/validate_features.py feature-list.json`.
