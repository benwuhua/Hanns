# HNSW Fairness Gate Datatype Screen Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local-only datatype capability audit for the HNSW fair lane so the project can decide whether harness-only BF16 requests are honest enough to promote.

**Architecture:** Extend the HDF5 baseline binary with a requested datatype option and explicit requested-vs-effective datatype metadata, then use a local HDF5 screen to see whether a BF16 request changes the real HNSW lane. Keep the current authority artifacts untouched; this slice ends with a local screen decision only.

**Tech Stack:** Rust 2021, `src/bin/generate_hdf5_hnsw_baseline.rs`, local `cargo test`, local `cargo run --release --features hdf5`, durable log in `task-progress.md`.

---

### Task 1: Tighten The Datatype Audit Contract

**Files:**
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`

- [ ] **Step 1: Write the failing test**

Add a focused unit test in the baseline binary that requests `bfloat16` for the HNSW lane and expects:

- requested datatype label = `BFloat16`
- effective datatype label = `Float32`

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `cargo test --features hdf5 --bin generate_hdf5_hnsw_baseline requested_bfloat16_reports_float32_effective_lane_for_current_hnsw -- --nocapture`

Expected: `FAIL` because the requested/effective datatype audit helpers do not exist yet.

### Task 2: Implement The Harness-Side Datatype Audit

**Files:**
- Modify: `src/bin/generate_hdf5_hnsw_baseline.rs`

- [ ] **Step 1: Add requested datatype parsing**

Add a small datatype parser for:

- `float32`
- `bfloat16`

and a CLI flag:

- `--vector-datatype <float32|bfloat16>`

Default remains `float32`.

- [ ] **Step 2: Add requested/effective datatype metadata helpers**

Implement helpers that:

- map the requested CLI value to the requested `DataType`
- report the requested datatype label
- report the effective current HNSW datatype label

For this screen, the current HNSW effective datatype remains `Float32`.

- [ ] **Step 3: Emit the new artifact metadata**

Extend the baseline output row to include:

- `requested_vector_datatype`

Keep the existing `vector_datatype` field as the effective datatype.

- [ ] **Step 4: Run the binary tests**

Run: `cargo test --features hdf5 --bin generate_hdf5_hnsw_baseline -- --nocapture`

Expected: `PASS`

### Task 3: Run The Local Datatype Screen

**Files:**
- Modify: none

- [ ] **Step 1: Run the current fair local lane with requested `float32`**

Run:

`cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/hnsw_fairness_datatype_float32.json --base-limit 100000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --hnsw-adaptive-k 0 --query-dispatch-mode parallel --query-batch-size 32 --vector-datatype float32 --recall-gate 0.95 --random-seed 42`

Expected: `ok`

- [ ] **Step 2: Run the same fair local lane with requested `bfloat16`**

Run:

`cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input data/sift/sift-128-euclidean.hdf5 --output /tmp/hnsw_fairness_datatype_bfloat16.json --base-limit 100000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --hnsw-adaptive-k 0 --query-dispatch-mode parallel --query-batch-size 32 --vector-datatype bfloat16 --recall-gate 0.95 --random-seed 42`

Expected: `ok`

- [ ] **Step 3: Compare the artifacts**

Verify that:

- requested `bfloat16` artifact reports `requested_vector_datatype = "BFloat16"`
- the same artifact still reports `vector_datatype = "Float32"`

If that mismatch remains, record `screen_result=reject` for the harness-only datatype path.

### Task 4: Durable Screen Handoff

**Files:**
- Modify: `task-progress.md`

- [ ] **Step 1: Record the local screen outcome**

Add a new session entry with:

- focus
- commands run
- result
- the exact reason the screen promoted or rejected

- [ ] **Step 2: Run final local verification**

Run:

- `cargo fmt --all -- --check`
- `cargo test --features hdf5 --bin generate_hdf5_hnsw_baseline -- --nocapture`
- `python3 scripts/validate_features.py feature-list.json`

Expected:

- all commands pass
- durable state remains `66/66`

- [ ] **Step 3: Commit only if the screen result is worth preserving**

If the screen gives a clear architectural result, commit the docs/code/task-progress changes with a focused message.
