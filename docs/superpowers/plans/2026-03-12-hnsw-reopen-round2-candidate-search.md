# HNSW Reopen Round 2 Candidate-Search Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reopen HNSW for a second candidate-search-centered algorithm line and judge it by fresh remote same-schema recall-gated QPS evidence.

**Architecture:** Keep the historical HNSW verdict chain intact, add round-2-specific reopen artifacts, and limit implementation changes to the shared candidate-search core in `src/faiss/hnsw.rs` plus the durable workflow files that describe and verify the new line. The first chunk only activates round 2; later chunks deepen profiling, rework the core, and rerun the authority lane.

**Tech Stack:** Rust 2021, cargo tests, remote x86 authority wrappers, JSON benchmark artifacts, durable markdown/json workflow state.

---

## Chunk 1: Activate Round 2

### Task 1: Add a failing round-2 activation regression

**Files:**
- Create: `tests/bench_hnsw_reopen_round2.rs`

- [ ] **Step 1: Write the failing test**

Write a default-lane regression that loads `benchmark_results/hnsw_reopen_round2_baseline.json` and asserts:

- `task_id == "HNSW-REOPEN-ROUND2-BASELINE"`
- `family == "HNSW"`
- `authority_scope == "remote_x86_only"`
- `historical_verdict_source == "benchmark_results/hnsw_p3_002_final_verdict.json"`
- `round1_baseline_source == "benchmark_results/hnsw_reopen_baseline.json"`
- `round1_profile_source == "benchmark_results/hnsw_reopen_profile_round1.json"`
- `round2_target == "candidate_search_same_schema_qps"`
- the summary mentions the unchanged `functional-but-not-leading` historical verdict

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round2 -- --nocapture`
Expected: FAIL because `benchmark_results/hnsw_reopen_round2_baseline.json` does not exist yet.

### Task 2: Add the round-2 baseline artifact and durable state

**Files:**
- Create: `benchmark_results/hnsw_reopen_round2_baseline.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Create the round-2 baseline artifact**

Write `benchmark_results/hnsw_reopen_round2_baseline.json` with:

- the round-1 authority refresh metrics copied from `benchmark_results/hnsw_reopen_profile_round1.json`
- references to the historical HNSW verdict and reopen baseline artifacts
- `round2_target` set to `candidate_search_same_schema_qps`
- a summary explaining that round 1 improved build time but not the HNSW family verdict

- [ ] **Step 2: Extend durable workflow state**

Add four new failing features to `feature-list.json`:

- `hnsw-reopen-round2-activation`
- `hnsw-candidate-search-profiler`
- `hnsw-candidate-search-core-rework`
- `hnsw-round2-authority-same-schema-rerun`

Mark only `hnsw-reopen-round2-activation` as the active feature for this chunk.

- [ ] **Step 3: Update progress and governance docs**

Update `task-progress.md`, `TASK_QUEUE.md`, `GAP_ANALYSIS.md`, `DEV_ROADMAP.md`, `RELEASE_NOTES.md`, and `docs/PARITY_AUDIT.md` so they describe:

- round 1 as closed with mixed evidence
- round 2 as active
- candidate search as the next explicit hypothesis

- [ ] **Step 4: Run the regression to verify it passes**

Run: `cargo test --test bench_hnsw_reopen_round2 -- --nocapture`
Expected: PASS

- [ ] **Step 5: Refresh remote sync and authority replay**

Run: `bash init.sh`
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round2 -q"`
Expected: `test=ok`

- [ ] **Step 6: Validate durable state and commit**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`

```bash
git add tests/bench_hnsw_reopen_round2.rs benchmark_results/hnsw_reopen_round2_baseline.json feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): activate reopen round 2"
```

## Chunk 2: Profile Candidate Search More Deeply

### Task 3: Add a failing round-2 candidate-search profiling contract

**Files:**
- Modify: `tests/bench_hnsw_reopen_round2.rs`
- Create: `tests/bench_hnsw_reopen_round2_profile.rs`

- [ ] **Step 1: Tighten the round-2 contract**

Extend `tests/bench_hnsw_reopen_round2.rs` so it also requires `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json` with buckets for:

- `entry_descent_ms`
- `frontier_ops_ms`
- `visited_ops_ms`
- `distance_compute_ms`
- `candidate_pruning_ms`

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round2 -- --nocapture`
Expected: FAIL because the new profile artifact does not exist yet.

### Task 4: Implement the round-2 profiler

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Create: `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Add profiling support in `src/faiss/hnsw.rs`**

Instrument the shared candidate-search core so the profiler can report the finer-grained buckets listed above without changing search semantics.

- [ ] **Step 2: Add the long-test generator**

Create `tests/bench_hnsw_reopen_round2_profile.rs` to run the profiler and emit `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json`.

- [ ] **Step 3: Generate the artifact locally**

Run: `cargo test --features long-tests --test bench_hnsw_reopen_round2_profile -- --ignored --nocapture`
Expected: PASS and writes `benchmark_results/hnsw_reopen_candidate_search_profile_round2.json`

- [ ] **Step 4: Re-run the contract test**

Run: `cargo test --test bench_hnsw_reopen_round2 -- --nocapture`
Expected: PASS

- [ ] **Step 5: Authority replay**

Run: `bash init.sh`
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round2_profile -- --ignored --nocapture"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round2 -q"`
Expected: `test=ok`

- [ ] **Step 6: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs tests/bench_hnsw_reopen_round2.rs tests/bench_hnsw_reopen_round2_profile.rs benchmark_results/hnsw_reopen_candidate_search_profile_round2.json feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): profile round 2 candidate search"
```

## Chunk 3: Rework the Shared Candidate-Search Core

### Task 5: Add focused candidate-search regressions

**Files:**
- Modify: `src/faiss/hnsw.rs`

- [ ] **Step 1: Add failing focused regressions**

Add targeted tests around:

- visited-state reset correctness
- scratch reuse without result contamination
- shared candidate expansion consistency between build and search
- queue/heap operations preserving the recall floor on deterministic fixtures

- [ ] **Step 2: Run focused library tests to verify failure**

Run: `cargo test hnsw --lib -- --nocapture`
Expected: FAIL on the new focused candidate-search regressions.

### Task 6: Implement the shared core rework

**Files:**
- Modify: `src/faiss/hnsw.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Implement the minimal core changes**

Rework scratch handling, visited resets, and queue behavior in the shared candidate-search core while preserving the external API and the current graph semantics.

- [ ] **Step 2: Re-run focused library tests**

Run: `cargo test hnsw --lib -- --nocapture`
Expected: PASS

- [ ] **Step 3: Re-run reopen contracts**

Run: `cargo test --test bench_hnsw_cpp_compare -q`
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_progress -q`
Expected: PASS

Run: `cargo test --test bench_hnsw_reopen_round2 -q`
Expected: PASS

- [ ] **Step 4: Authority safety replay**

Run: `bash init.sh`
Expected: PASS

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_cpp_compare -q"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round2 -q"`
Expected: `test=ok`

- [ ] **Step 5: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`

```bash
git add src/faiss/hnsw.rs feature-list.json task-progress.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "feat(hnsw): rework round 2 candidate search core"
```

## Chunk 4: Refresh the Real Authority Lane

### Task 7: Re-run same-schema authority and summarize round 2

**Files:**
- Create: `benchmark_results/hnsw_reopen_round2_authority_summary.json`
- Modify: `tests/bench_hnsw_reopen_round2.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Tighten the round-2 contract again**

Extend `tests/bench_hnsw_reopen_round2.rs` so it requires `benchmark_results/hnsw_reopen_round2_authority_summary.json` and asserts:

- the recorded round-2 target is `candidate_search_same_schema_qps`
- the summary includes recall/QPS deltas versus the reopen baseline
- the artifact explicitly states whether HNSW may rewrite the historical family verdict

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --test bench_hnsw_reopen_round2 -- --nocapture`
Expected: FAIL because the authority summary artifact does not exist yet.

- [ ] **Step 3: Refresh remote sync**

Run: `bash init.sh`
Expected: PASS

- [ ] **Step 4: Re-run the authoritative same-schema lane**

Run the recorded remote same-schema HNSW compare/benchmark command needed to regenerate the current HNSW authority result. Save the resulting log path in the round-2 summary artifact.

- [ ] **Step 5: Re-run round-2 profile and contract lanes on authority**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --features long-tests --test bench_hnsw_reopen_round2_profile -- --ignored --nocapture"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round2 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round2 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round2 -q"`
Expected: `test=ok`

- [ ] **Step 6: Write the round-2 authority summary**

Create `benchmark_results/hnsw_reopen_round2_authority_summary.json` with:

- fresh same-schema recall/QPS result
- deltas versus `benchmark_results/hnsw_reopen_baseline.json`
- references to the round-2 profile artifact and authority logs
- `verdict_refresh_allowed` as `true` or `false`
- `next_action` as `continue`, `soft_stop`, or `hard_stop`

- [ ] **Step 7: Re-run the contract test**

Run: `cargo test --test bench_hnsw_reopen_round2 -- --nocapture`
Expected: PASS

- [ ] **Step 8: Update durable workflow state**

If `verdict_refresh_allowed == true`:

- update the HNSW family verdict chain and reopen docs accordingly

If `verdict_refresh_allowed == false`:

- keep the historical family verdict intact
- record whether round 2 ends in continue, soft stop, or hard stop

- [ ] **Step 9: Validate and commit**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID`

```bash
git add benchmark_results/hnsw_reopen_round2_authority_summary.json tests/bench_hnsw_reopen_round2.rs feature-list.json task-progress.md TASK_QUEUE.md GAP_ANALYSIS.md DEV_ROADMAP.md RELEASE_NOTES.md docs/PARITY_AUDIT.md
git commit -m "test(hnsw): refresh round 2 authority summary"
```
