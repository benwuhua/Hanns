# DiskANN Remote Benchmark Gate Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the DiskANN remote benchmark gate with a durable benchmark-gate artifact, real default-lane regressions, and refreshed authority-backed cross-dataset evidence.

**Architecture:** Keep this feature narrower than the final DiskANN family verdict. First, add failing default-lane regressions that prove the benchmark-gate artifact is missing and that the current cross-dataset artifact does not yet cover DiskANN. Then add the benchmark-gate artifact plus DiskANN support in `src/benchmark/cross_dataset_sampling.rs`, refresh the changed artifact on the remote authority host, and finally sync the durable workflow state once the recorded remote verification commands pass.

**Tech Stack:** Rust 2021 integration tests in `tests/`, benchmark artifact generators in `src/benchmark/`, JSON artifacts in `benchmark_results/`, remote authority wrappers in `scripts/remote/`, durable workflow files (`feature-list.json`, `task-progress.md`, `RELEASE_NOTES.md`, `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, `docs/PARITY_AUDIT.md`, `DEV_ROADMAP.md`)

---

## Chunk 1: Lock The Benchmark Gate Contract With Failing Tests

### Task 1: Add default-lane DiskANN regressions for the benchmark gate

**Files:**
- Modify: `tests/bench_diskann_1m.rs`
- Modify: `tests/bench_recall_gated_baseline.rs`
- Modify: `tests/bench_cross_dataset_sampling.rs`

- [ ] **Step 1: Write the failing benchmark-gate artifact regression**

Add a default-lane test in `tests/bench_diskann_1m.rs` that expects `benchmark_results/diskann_p3_004_benchmark_gate.json` to exist and to declare `benchmark_gate_verdict = no_go_for_native_comparable_benchmark` with `comparability_status = constrained`.

- [ ] **Step 2: Write the baseline-artifact DiskANN regression**

Add a default-lane test in `tests/bench_recall_gated_baseline.rs` that loads the current baseline artifact and asserts the DiskANN row exists, stays below the recall gate, and is not `trusted`.

- [ ] **Step 3: Write the cross-dataset DiskANN regression**

Add a default-lane test in `tests/bench_cross_dataset_sampling.rs` that expects DiskANN rows for every sampled dataset and asserts each row remains below the recall gate or otherwise non-trusted.

- [ ] **Step 4: Run the focused tests to verify RED**

Run: `cargo test --test bench_diskann_1m -- --nocapture`
Expected: FAIL because the benchmark-gate artifact does not exist yet.

Run: `cargo test --test bench_recall_gated_baseline -- --nocapture`
Expected: PASS or FAIL only after the new DiskANN assertions land; either way the file must now execute a real DiskANN regression.

Run: `cargo test --test bench_cross_dataset_sampling -- --nocapture`
Expected: FAIL because `cross_dataset_sampling.json` does not yet include DiskANN rows.

## Chunk 2: Add DiskANN Cross-Dataset Coverage And The Gate Artifact

### Task 2: Implement the minimal benchmark-gate closure

**Files:**
- Create: `benchmark_results/diskann_p3_004_benchmark_gate.json`
- Modify: `src/benchmark/cross_dataset_sampling.rs`
- Modify: `tests/bench_diskann_1m.rs`
- Modify: `tests/bench_recall_gated_baseline.rs`
- Modify: `tests/bench_cross_dataset_sampling.rs`

- [ ] **Step 1: Add DiskANN rows to the cross-dataset generator**

Extend `src/benchmark/cross_dataset_sampling.rs` with a constrained DiskANN benchmark path so each sampled dataset emits a DiskANN row and validation expects `3 datasets x 4 indexes`.

- [ ] **Step 2: Add the benchmark-gate artifact**

Create `benchmark_results/diskann_p3_004_benchmark_gate.json` that summarizes the current benchmark-gate outcome using the baseline artifact, the refreshed cross-dataset artifact, and the constrained-scope guard.

- [ ] **Step 3: Refresh the cross-dataset artifact locally only as needed for iteration**

Use the existing long-test artifact generator to get the local repo into a self-consistent state before rerunning the focused regressions.

- [ ] **Step 4: Run the focused tests to verify GREEN**

Run: `cargo test --test bench_diskann_1m -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_recall_gated_baseline -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_cross_dataset_sampling -- --nocapture`
Expected: PASS

## Chunk 3: Refresh Authority Evidence And Sync Durable State

### Task 3: Close the feature honestly in the long-task workflow

**Files:**
- Modify: `benchmark_results/cross_dataset_sampling.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `DEV_ROADMAP.md`

- [ ] **Step 1: Refresh the changed benchmark artifact on the authority host**

Run: `bash init.sh`
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-diskann-benchmark-gate KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-diskann-benchmark-gate bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_cross_dataset_sampling -- --ignored --nocapture"`
Expected: `test=ok`

Run: use `scripts/remote/common.sh` rsync helpers to copy the refreshed `benchmark_results/cross_dataset_sampling.json` from the authority repo back into the local repo.
Expected: local artifact now contains DiskANN rows produced by the authority run.

- [ ] **Step 2: Run the recorded local verification commands**

Run: `cargo test --test bench_diskann_1m -q`
Expected: PASS

Run: `cargo test --test bench_recall_gated_baseline -q`
Expected: PASS

Run: `cargo test --test bench_cross_dataset_sampling -q`
Expected: PASS

- [ ] **Step 3: Run the recorded remote verification commands**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-diskann-benchmark-gate KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-diskann-benchmark-gate bash scripts/remote/test.sh --command "cargo test --test bench_diskann_1m -q"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-diskann-benchmark-gate KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-diskann-benchmark-gate bash scripts/remote/test.sh --command "cargo test --test bench_recall_gated_baseline -q"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-diskann-benchmark-gate KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-diskann-benchmark-gate bash scripts/remote/test.sh --command "cargo test --test bench_cross_dataset_sampling -q"`
Expected: `test=ok`

- [ ] **Step 4: Update durable state**

Mark `diskann-remote-benchmark-gate` as `passing`, record the benchmark-gate artifact and authority logs in the durable docs, set the next feature handoff, and re-run `python3 scripts/validate_features.py feature-list.json`.
