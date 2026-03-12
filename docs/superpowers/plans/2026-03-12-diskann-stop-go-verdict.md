# DiskANN Stop-Go Verdict Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Archive a family-level DiskANN final verdict that classifies the family as constrained while keeping benchmark and compare-lane no-go boundaries mechanically enforced.

**Architecture:** Add one family-level final verdict artifact and bind it to the three recorded verification entrypoints already listed in `feature-list.json`. Library tests will lock the verdict to `DiskAnnIndex` and `PQFlashIndex` scope-audit facts, while the benchmark and compare lanes will assert that the final verdict stays aligned with the benchmark-gate artifact and continued compare-lane exclusion.

**Tech Stack:** Rust 2021 library/unit tests in `src/faiss/`, integration tests in `tests/`, JSON verdict artifacts in `benchmark_results/`, remote authority wrappers in `scripts/remote/`, durable workflow files in the repo root and `docs/`

---

## Chunk 1: Add Failing Final-Verdict Regressions

### Task 1: Make the recorded verification surfaces fail for the missing family verdict

**Files:**
- Modify: `src/faiss/diskann.rs`
- Modify: `tests/bench_diskann_1m.rs`
- Modify: `tests/bench_compare.rs`

- [ ] **Step 1: Add a failing library regression for the final verdict artifact**

In `src/faiss/diskann.rs`, add a unit test that loads `benchmark_results/diskann_p3_004_final_verdict.json` and asserts:

- `family == "DiskANN"`
- `classification == "constrained"`
- `leadership_verdict == "no_go_for_native_comparable_benchmark"`
- `leadership_claim_allowed == false`

- [ ] **Step 2: Add a failing benchmark-lane regression for the final verdict**

In `tests/bench_diskann_1m.rs`, add a default-lane test that loads the final verdict artifact and asserts the benchmark-gate and final-verdict artifacts agree on:

- family name
- leadership/no-go benchmark posture
- `native_comparison_allowed == false`

- [ ] **Step 3: Add a failing compare-lane regression for the final verdict**

In `tests/bench_compare.rs`, add a default-lane test that loads the final verdict artifact and asserts:

- `classification == "constrained"`
- `leadership_claim_allowed == false`
- compare lane still excludes `DiskANN`

- [ ] **Step 4: Run focused tests to verify RED**

Run: `cargo test --lib diskann -- --nocapture`
Expected: FAIL because the final verdict artifact does not exist yet.

Run: `cargo test --test bench_diskann_1m -- --nocapture`
Expected: FAIL because the final verdict artifact does not exist yet.

Run: `cargo test --test bench_compare -- --nocapture`
Expected: FAIL because the final verdict artifact does not exist yet.

## Chunk 2: Add The Family Verdict And Make The Regressions Pass

### Task 2: Implement the minimum final-verdict closure

**Files:**
- Create: `benchmark_results/diskann_p3_004_final_verdict.json`
- Modify: `src/faiss/diskann.rs`
- Modify: `tests/bench_diskann_1m.rs`
- Modify: `tests/bench_compare.rs`

- [ ] **Step 1: Create the family-level final verdict artifact**

Add `benchmark_results/diskann_p3_004_final_verdict.json` with:

- `task_id = DISKANN-P3-004`
- `family = DiskANN`
- `classification = constrained`
- `leadership_verdict = no_go_for_native_comparable_benchmark`
- `leadership_claim_allowed = false`
- references to the benchmark-gate, baseline, cross-dataset, and scope/contract evidence

- [ ] **Step 2: Tighten the library regression around executable boundary facts**

Keep the new library regression narrow: it should assert both the artifact classification and the scope-audit facts that justify it, rather than inventing new production code.

- [ ] **Step 3: Run focused tests to verify GREEN**

Run: `cargo test --lib diskann -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_diskann_1m -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_compare -- --nocapture`
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

Run: `cargo test --lib diskann -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_diskann_1m -q`
Expected: PASS

Run: `cargo test --test bench_compare -q`
Expected: PASS

- [ ] **Step 2: Refresh remote authority sync and run the recorded remote verification commands**

Run: `bash init.sh`
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-diskann-stop-go KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-diskann-stop-go bash scripts/remote/test.sh --command "cargo test --lib diskann -- --nocapture"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-diskann-stop-go KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-diskann-stop-go bash scripts/remote/test.sh --command "cargo test --test bench_diskann_1m -q"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-diskann-stop-go KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-diskann-stop-go bash scripts/remote/test.sh --command "cargo test --test bench_compare -q"`
Expected: `test=ok`

- [ ] **Step 3: Update durable workflow state**

Mark `diskann-stop-go-verdict` as `passing`, record the new final verdict artifact and authority logs in durable docs, set the next feature handoff, and re-run `python3 scripts/validate_features.py feature-list.json`.
