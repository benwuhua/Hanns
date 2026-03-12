# HNSW Reopen Algorithm Line Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reopen HNSW as an active algorithm-improvement line, freeze a new reopen baseline, and ship the first core build-path rework with fresh remote x86 evidence.

**Architecture:** Keep the 2026-03-12 final verdict artifacts intact as historical baseline, then add a parallel HNSW reopen line with its own progress artifacts and tests. The first real algorithm slice should target the current build-path inconsistency in `src/faiss/hnsw.rs`: bulk build still uses a weaker connection-update path than single-node insertion, and build-time layer search still leaves scratch-friendly hot paths underused. The reopen line therefore needs three layers: durable-state reopening, progress/profiler surfaces, and one concrete build-quality rework tied to the authority benchmark lane.

**Tech Stack:** Rust 2021 library and integration tests, JSON progress artifacts in `benchmark_results/`, durable workflow docs in repo-root markdown/JSON, remote authority wrappers in `scripts/remote/`

---

## File Map

- Create: `benchmark_results/hnsw_reopen_baseline.json`
- Create: `benchmark_results/hnsw_reopen_profile_round1.json`
- Create: `tests/bench_hnsw_reopen_progress.rs`
- Create: `tests/bench_hnsw_reopen_profile.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `src/faiss/hnsw.rs`
- Reuse for safety checks: `tests/bench_hnsw_cpp_compare.rs`

## Chunk 1: Reopen HNSW Durable State And Freeze The Baseline

### Task 1: Add a failing default-lane reopen baseline regression

**Files:**
- Create: `tests/bench_hnsw_reopen_progress.rs`

- [ ] **Step 1: Write the failing baseline-artifact regression**

Add a default-lane test in `tests/bench_hnsw_reopen_progress.rs` that loads `benchmark_results/hnsw_reopen_baseline.json` and asserts:

- `family == "HNSW"`
- `authority_scope == "remote_x86_only"`
- `historical_verdict_source == "benchmark_results/hnsw_p3_002_final_verdict.json"`
- baseline evidence records the currently archived Rust/native QPS and recall values

- [ ] **Step 2: Run the test to verify RED**

Run: `cargo test --test bench_hnsw_reopen_progress -- --nocapture`
Expected: FAIL because `benchmark_results/hnsw_reopen_baseline.json` does not exist yet.

### Task 2: Add the baseline artifact and reopen the workflow state

**Files:**
- Create: `benchmark_results/hnsw_reopen_baseline.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Create `benchmark_results/hnsw_reopen_baseline.json`**

Record the current trusted baseline:

- historical HNSW family verdict source
- current same-schema Rust recall/QPS
- current same-schema native recall/QPS
- current native-over-Rust gap ratio
- reopen status set to `active`

- [ ] **Step 2: Reopen HNSW in durable workflow state**

Update durable files so:

- HNSW is the only active reopened line
- `task-progress.md` no longer shows `Current focus: none`
- `feature-list.json` gains the first reopen features:
  - `hnsw-reopen-baseline-freeze`
  - `hnsw-build-path-profiler`
  - `hnsw-build-quality-rework`
  - `hnsw-authority-rerun-and-verdict-refresh`
- historical final artifacts remain documented as baseline truth, not deleted

- [ ] **Step 3: Run the test to verify GREEN**

Run: `cargo test --test bench_hnsw_reopen_progress -- --nocapture`
Expected: PASS

- [ ] **Step 4: Validate reopened durable state**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`

- [ ] **Step 5: Commit the reopen baseline slice**

```bash
git add benchmark_results/hnsw_reopen_baseline.json tests/bench_hnsw_reopen_progress.rs feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): reopen algorithm improvement line"
```

## Chunk 2: Add A Real HNSW Reopen Profiling Surface

### Task 3: Make profiling progress mechanically visible

**Files:**
- Modify: `tests/bench_hnsw_reopen_progress.rs`
- Create: `tests/bench_hnsw_reopen_profile.rs`

- [ ] **Step 1: Extend the progress regression to require a profile artifact**

Add a second test in `tests/bench_hnsw_reopen_progress.rs` that loads `benchmark_results/hnsw_reopen_profile_round1.json` and asserts it contains:

- a named benchmark lane tied to the HNSW reopen line
- build-path timing buckets
- explicit hotspot ranking
- a recommended first rework target

- [ ] **Step 2: Add a long-test profile generator shell**

Create `tests/bench_hnsw_reopen_profile.rs` with a `long-tests` + `#[ignore]` generator that:

- loads the same SIFT1M/HDF5-compatible dataset surface already used by HNSW compare work
- runs HNSW build/search with instrumentation enabled
- writes `benchmark_results/hnsw_reopen_profile_round1.json`

- [ ] **Step 3: Run the default-lane regression to verify RED**

Run: `cargo test --test bench_hnsw_reopen_progress -- --nocapture`
Expected: FAIL because `benchmark_results/hnsw_reopen_profile_round1.json` does not exist yet.

### Task 4: Instrument `src/faiss/hnsw.rs` for build-path profiling

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `benchmark_results/hnsw_reopen_profile_round1.json`
- Modify: `tests/bench_hnsw_reopen_progress.rs`
- Modify: `tests/bench_hnsw_reopen_profile.rs`

- [ ] **Step 1: Add lightweight build-path stats structures**

In `src/faiss/hnsw.rs`, add a minimal profiling structure that can report:

- time spent in layer descent / candidate search
- time spent in neighbor selection
- time spent in reverse-link updates / shrink
- time spent in repair
- call counts for the same stages

Keep the profiling entrypoint separate from normal search/build APIs unless a shared helper is clearly reusable.

- [ ] **Step 2: Use the instrumentation in the profile generator**

Make `tests/bench_hnsw_reopen_profile.rs` emit a concrete JSON artifact with the stats above plus a ranked list of top hotspots.

- [ ] **Step 3: Run local tests to verify GREEN**

Run: `cargo test --test bench_hnsw_reopen_progress -- --nocapture`
Expected: PASS

Run: `cargo test --features long-tests --test bench_hnsw_reopen_profile -- --ignored --nocapture`
Expected: PASS and rewrites `benchmark_results/hnsw_reopen_profile_round1.json`

- [ ] **Step 4: Commit the profiling slice**

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_progress.rs tests/bench_hnsw_reopen_profile.rs benchmark_results/hnsw_reopen_profile_round1.json
git commit -m "feat(hnsw): add reopen build-path profiler"
```

## Chunk 3: Ship The First Core Build-Quality Rework

### Task 5: Expose the current bulk-build path inconsistency with failing tests

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add a failing regression for build-path connection semantics**

In `src/faiss/hnsw.rs`, add a focused test that demonstrates the current bulk-build path does not use the same shrink/diversification behavior as single-node insertion. The test should be deterministic and should compare:

- one index built through the bulk `add` path
- one index built through repeated single-node insertion

The assertion should focus on graph-structure semantics, not raw performance numbers. Examples:

- layer-0 degree underfill on the bulk path
- reverse-link shrink behavior diverging between the two build paths
- connectivity/neighbor-count drift that should not exist after the rework

- [ ] **Step 2: Run the focused unit test to verify RED**

Run: `cargo test hnsw --lib -- --nocapture`
Expected: FAIL on the new regression.

### Task 6: Rework the first HNSW hot path

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Replace the weaker bulk-build connection update path**

Refactor the build path so the bulk insertion flow no longer relies on the weaker `add_connections_for_node()` semantics. Reuse the stronger connection maintenance path used by single-node insertion where practical, including heuristic shrink behavior.

- [ ] **Step 2: Reuse scratch-friendly search helpers inside build**

Move build-time layer search toward `search_layer_idx_with_scratch()` or an equivalent reusable scratch-based path so repeated insertion/build work stops paying avoidable allocation costs in the hot loop.

- [ ] **Step 3: Keep API/FFI behavior unchanged**

Do not change public HNSW search APIs, persistence semantics, or the FFI metadata contracts during this step.

- [ ] **Step 4: Run focused local verification**

Run: `cargo test hnsw --lib -- --nocapture`
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_progress -q`
Expected: PASS

Run: `cargo test --test bench_hnsw_cpp_compare -q`
Expected: PASS

- [ ] **Step 5: Commit the first algorithm slice**

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_progress.rs
git commit -m "feat(hnsw): rework bulk build path"
```

## Chunk 4: Refresh Authority Evidence And Decide Whether HNSW Moves

### Task 7: Re-run the authority lane after the first rework

**Files:**
- Modify: `benchmark_results/hnsw_reopen_profile_round1.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`
- Modify: `feature-list.json`

- [ ] **Step 1: Refresh remote sync**

Run: `bash init.sh`
Expected: PASS

- [ ] **Step 2: Re-run the reopen profile on authority**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_profile -- --ignored --nocapture"`
Expected: `test=ok`

- [ ] **Step 3: Re-run the existing compare lane on authority**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"`
Expected: `test=ok`

- [ ] **Step 4: Re-run the new reopen progress lane on authority**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_progress -q"`
Expected: `test=ok`

- [ ] **Step 5: Update durable state with the result**

If authority evidence improves:

- record the new Rust metrics relative to the reopen baseline
- keep HNSW active and hand off to the next HNSW rework feature

If authority evidence does not improve:

- record the failed hypothesis explicitly in the reopen docs
- decide whether to continue with a second targeted HNSW rework or stop after two consecutive non-improving iterations

- [ ] **Step 6: Validate and commit the authority refresh**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`

```bash
git add benchmark_results/hnsw_reopen_profile_round1.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "test(hnsw): refresh reopen authority evidence"
```
