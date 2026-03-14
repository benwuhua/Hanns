# HNSW Reopen Round 12 Activation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the local Task 16 `shared_bitset_batch4` screen win into tracked authority-grade work by freezing a round-12 baseline, adding the activation contract, and reopening `feature-list.json`.

**Architecture:** Mirror the earlier round-11 activation pattern rather than inventing a new workflow. This slice does not run the authority same-schema benchmark yet; it only creates the round-12 default-lane contract plus baseline artifact, reopens the durable tracked state with one passing activation feature and one new failing authority feature, and verifies that both local and remote default-lane checks accept the new activation baseline.

**Tech Stack:** Rust 2021, JSON artifacts under `benchmark_results/`, default-lane regression in `tests/`, durable workflow files (`feature-list.json`, `task-progress.md`, `RELEASE_NOTES.md`, `docs/PARITY_AUDIT.md`), local plus remote verification through `bash init.sh` and `scripts/remote/test.sh`.

---

## File Map

- Create: `docs/superpowers/plans/2026-03-14-hnsw-reopen-round12-activation.md`
  - Focused execution plan for reopening HNSW around the promoted Task 16 mechanism.
- Create: `tests/bench_hnsw_reopen_round12.rs`
  - Default-lane contract for the round-12 baseline artifact and the future authority summary artifact.
- Create: `benchmark_results/hnsw_reopen_round12_baseline.json`
  - Activation baseline freezing the round-11 authority hard stop and the promoted Task-16 local screen evidence.
- Modify: `feature-list.json`
  - Add `hnsw-reopen-round12-activation` as passing after verification and `hnsw-round12-authority-same-schema-rerun` as the next failing tracked feature.
- Modify: `task-progress.md`
  - Record the activation work, commands, and handoff to the round-12 authority rerun.
- Modify: `RELEASE_NOTES.md`
  - Record the tracked round-12 activation.
- Modify: `docs/PARITY_AUDIT.md`
  - Record the activation rationale and the next authority question.

## Chunk 1: Lock the Round-12 Activation Contract

### Task 1: Add the default-lane round-12 activation test

**Files:**
- Create: `tests/bench_hnsw_reopen_round12.rs`

- [ ] **Step 1: Write the failing activation contract**

Create a new default-lane regression that:

- requires `benchmark_results/hnsw_reopen_round12_baseline.json`
- validates the round-12 target is `shared_bitset_batch4`
- validates the baseline references:
  - `benchmark_results/hnsw_p3_002_final_verdict.json`
  - `benchmark_results/hnsw_reopen_round11_authority_summary.json`
  - `benchmark_results/hnsw_bitset_search_cost_diagnosis.json`
- includes a future authority-summary check for `benchmark_results/hnsw_reopen_round12_authority_summary.json`, but marks it `#[ignore]` during activation

- [ ] **Step 2: Run the contract to verify red**

Run:

```bash
cargo test --test bench_hnsw_reopen_round12 -- --nocapture
```

Expected: FAIL because the round-12 baseline artifact does not exist yet.

- [ ] **Step 3: Add the minimal activation baseline artifact**

Create `benchmark_results/hnsw_reopen_round12_baseline.json` with:

- round-11 authority hard-stop context
- Task-16 promoted local bitset-lane context
- `round12_target = "shared_bitset_batch4"`
- a summary explaining that round 12 is reopening HNSW around the promoted shared bitset batch-4 screen signal while keeping the historical HNSW family verdict unchanged until authority evidence arrives

- [ ] **Step 4: Re-run the activation contract**

Run:

```bash
cargo test --test bench_hnsw_reopen_round12 -- --nocapture
```

Expected: PASS, with the future authority-summary check ignored.

## Chunk 2: Reopen the Durable Workflow State

### Task 2: Add round-12 tracked features and activation notes

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Add round-12 tracked features**

Update `feature-list.json` to append:

- `hnsw-reopen-round12-activation`
  - `status: passing`
  - verification steps centered on `bash init.sh`, the new default-lane test, remote replay of that test, and feature validation
- `hnsw-round12-authority-same-schema-rerun`
  - `status: failing`
  - verification steps centered on remote same-schema rerun, native capture, remote replay of the round-12 contract, and feature validation

Keep the dependency edge:

- `hnsw-reopen-round12-activation` depends on `hnsw-round11-authority-same-schema-rerun`
- `hnsw-round12-authority-same-schema-rerun` depends on `hnsw-reopen-round12-activation`

- [ ] **Step 2: Record the activation in durable docs**

Update:

- `task-progress.md` with a new activation session
- `RELEASE_NOTES.md` with the new round-12 activation note
- `docs/PARITY_AUDIT.md` with the activation rationale and the next authority question

Expected durable outcome:

- round-12 activation is recorded as the completed tracked feature for this turn
- round-12 authority same-schema rerun is recorded as the next tracked feature

## Chunk 3: Verify and Close the Activation Feature

### Task 3: Run local and remote activation verification

**Files:**
- No additional file edits expected

- [ ] **Step 1: Run local activation checks**

Run:

```bash
cargo test --test bench_hnsw_reopen_round12 -- --nocapture
python3 scripts/validate_features.py feature-list.json
```

Expected: PASS.

- [ ] **Step 2: Run remote activation checks**

Run:

```bash
bash init.sh
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-reopen-round12 KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-reopen-round12 bash scripts/remote/test.sh --command "cargo test --test bench_hnsw_reopen_round12 -q"
```

Expected: PASS, proving the default-lane activation baseline survives the authority bootstrap path before any round-12 same-schema rerun.

- [ ] **Step 3: Final validation pass**

Run:

```bash
python3 scripts/validate_features.py feature-list.json
```

Expected: `VALID`, with the new activation feature passing and the round-12 authority rerun remaining the next failing tracked feature.
