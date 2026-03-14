# HNSW Round 12 Authority Same-Schema Rerun Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the final failing tracked feature by rerunning the round-12 authority same-schema lane, archiving a durable round-12 authority summary, and updating workflow state based only on remote x86 evidence.

**Architecture:** Follow the established round-10/11 reopen closure pattern rather than inventing new HNSW workflow. First tighten the round-12 default contract so the missing authority summary fails locally, then run the isolated round-12 authority commands, sync the fresh Rust artifact back into the local worktree, write the round-12 authority summary, and finally replay the contract plus validator before marking the feature passing.

**Tech Stack:** Rust 2021, existing HNSW reopen tests in `tests/`, JSON durability in `benchmark_results/`, remote authority wrappers in `scripts/remote/`, native log parsing via `src/bin/native_benchmark_qps_parser.rs`.

---

## File Map

- Modify: `tests/bench_hnsw_reopen_round12.rs`
  - Turn the future authority-summary check into the active round-12 default-lane contract.
- Modify: `benchmark_results/rs_hnsw_sift128.full_k100.json`
  - Refresh the local Rust same-schema artifact from the remote round-12 authority run.
- Create: `benchmark_results/hnsw_reopen_round12_authority_summary.json`
  - Freeze the round-12 same-schema authority result and decision.
- Modify: `feature-list.json`
  - Mark `hnsw-round12-authority-same-schema-rerun` passing only after all verification succeeds.
- Modify: `task-progress.md`
  - Record the authority commands, logs, and final round-12 outcome.
- Modify: `RELEASE_NOTES.md`
  - Record the round-12 authority closure.
- Modify: `docs/PARITY_AUDIT.md`
  - Append the round-12 authority evidence and next-action reasoning.

## Chunk 1: Tighten the Contract Before Authority Work

### Task 1: Make the missing round-12 authority summary fail on the default lane

**Files:**
- Modify: `tests/bench_hnsw_reopen_round12.rs`

- [ ] **Step 1: Write the failing authority-summary contract**

Remove the activation-only `#[ignore]` and tighten the round-12 summary assertions to match round-11 closure rules:

- `next_action` must be `continue`, `soft_stop`, or `hard_stop`
- `same_schema_current.rust_qps` must be numeric for completed runs or null only for explicit timeout/abort outcomes
- `same_schema_current.native_qps` must be numeric
- `summary` must mention the same-schema qps outcome

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_reopen_round12 -- --nocapture
```

Expected: FAIL because `benchmark_results/hnsw_reopen_round12_authority_summary.json` does not exist yet.

## Chunk 2: Run the Authority Evidence Chain

### Task 2: Refresh round-12 same-schema artifacts on the authority machine

**Files:**
- Modify: `benchmark_results/rs_hnsw_sift128.full_k100.json`

- [ ] **Step 1: Bootstrap authority**

Run:

```bash
bash init.sh
```

- [ ] **Step 2: Run the Rust same-schema benchmark on the isolated round-12 lane**

Run:

```bash
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round12 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round12 \
bash scripts/remote/test.sh --command "cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.full_k100.json --base-limit 1000000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --recall-gate 0.95"
```

Record the returned log path and whether the run completed or failed.

- [ ] **Step 3: Capture fresh native authority evidence**

Run:

```bash
bash scripts/remote/native_hnsw_qps_capture.sh --log-dir /data/work/knowhere-rs-logs-hnsw-reopen-round12 --gtest-filter Benchmark_float_qps.TEST_HNSW
```

Record the returned log path.

- [ ] **Step 4: Sync the Rust same-schema artifact back into the local worktree**

Use the remote sync helper or `rsync` over the configured SSH target to copy:

```bash
/data/work/knowhere-rs-src/benchmark_results/rs_hnsw_sift128.full_k100.json
```

back to local:

```bash
benchmark_results/rs_hnsw_sift128.full_k100.json
```

Expected: local artifact now matches the remote round-12 run if the Rust command completed.

## Chunk 3: Freeze the Authority Verdict and Close Durable State

### Task 3: Write the round-12 authority summary and verify green

**Files:**
- Create: `benchmark_results/hnsw_reopen_round12_authority_summary.json`
- Modify: `tests/bench_hnsw_reopen_round12.rs`
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Create the authority summary artifact**

Write `benchmark_results/hnsw_reopen_round12_authority_summary.json` with:

- `task_id = "HNSW-REOPEN-ROUND12-AUTHORITY-SUMMARY"`
- `round12_target = "shared_bitset_batch4"`
- references to `benchmark_results/hnsw_reopen_round12_baseline.json` and `benchmark_results/hnsw_p3_002_final_verdict.json`
- `same_schema_sources.rust = "benchmark_results/rs_hnsw_sift128.full_k100.json"`
- `same_schema_sources.native_log` set to the fresh native log path
- `authority_logs` containing the fresh Rust and native log paths plus any remote replay log
- `same_schema_current` populated from fresh authority evidence
- `delta_vs_round11_authority_baseline`
- `verdict_refresh_allowed`
- `next_action`
- `summary`
- `action`

- [ ] **Step 2: Re-run the local contract**

Run:

```bash
cargo test --test bench_hnsw_reopen_round12 -- --nocapture
```

Expected: PASS.

- [ ] **Step 3: Replay the default-lane contract on authority**

Run:

```bash
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round12 \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round12 \
bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round12 -q"
```

Expected: `test=ok`.

- [ ] **Step 4: Mark the feature and update durable docs**

Update:

- `feature-list.json`
- `task-progress.md`
- `RELEASE_NOTES.md`
- `docs/PARITY_AUDIT.md`

Set:

- `hnsw-round12-authority-same-schema-rerun` -> `passing`
- `Current focus: none`
- `Next feature: none`
- progress to `66/66`

- [ ] **Step 5: Run final validator**

Run:

```bash
python3 scripts/validate_features.py feature-list.json
```

Expected: `VALID - 66 features (66 passing, 0 failing); workflow/doc checks passed`.
