# HNSW Fairness Gate Effective Ef Authority Rerun Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the remote same-schema HNSW evidence chain with `--hnsw-adaptive-k 0` so the fair lane records matched effective `ef` while keeping datatype and query-dispatch blockers visible.

**Architecture:** Tighten the fairness artifact contract first, then regenerate the authoritative Rust HDF5 baseline on the remote x86 host with adaptive ef disabled. Pull the fresh Rust artifact back into the repo, rebuild the same-schema and fairness-gate JSON summaries from that authoritative row, and record the authority result in durable docs without claiming the overall fairness gate has passed.

**Tech Stack:** Rust 2021, Python unittest/json tooling, `src/bin/generate_hdf5_hnsw_baseline.rs`, remote wrappers in `scripts/remote/`, durable JSON artifacts in `benchmark_results/`, long-task docs in `task-progress.md` and `RELEASE_NOTES.md`.

---

### Task 1: Tighten The Fairness Contract

**Files:**
- Modify: `tests/test_hnsw_fairness_gate.py`

- [ ] **Step 1: Write the failing test**

Change the fairness gate contract so the authoritative fair-lane refresh must satisfy:
- `rust_row["adaptive_k"] == 0.0`
- `rust_row["effective_ef_search"] == rust_row["requested_ef_search"]`
- `fairness["checks"]["effective_ef_alignment"]["pass"] is True`

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m unittest tests/test_hnsw_fairness_gate.py`
Expected: `FAIL` because the committed authority artifacts still report `adaptive_k=2.0` and `effective_ef_search=200`.

### Task 2: Rebuild The Authority Rust Artifact

**Files:**
- Modify: `benchmark_results/rs_hnsw_sift128.full_k100.json`

- [ ] **Step 1: Bootstrap authority host**

Run: `bash init.sh`
Expected: workspace sync completes and remote x86 config is printed.

- [ ] **Step 2: Re-run the remote Rust same-schema baseline on the fair lane**

Run:
`KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-hnsw-fairness-effective-ef KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-hnsw-fairness-effective-ef bash scripts/remote/test.sh --command "cargo run --release --features hdf5 --bin generate_hdf5_hnsw_baseline -- --input /data/work/knowhere-native-src/sift-128-euclidean.hdf5 --output benchmark_results/rs_hnsw_sift128.full_k100.json --base-limit 1000000 --query-limit 1000 --top-k 100 --recall-at 10 --m 16 --ef-construction 100 --ef-search 138 --hnsw-adaptive-k 0 --recall-gate 0.95"`
Expected: `test=ok` with a retained remote log path under `/data/work/knowhere-rs-logs-hnsw-fairness-effective-ef/`.

- [ ] **Step 3: Verify the refreshed Rust artifact was synced back**

Check `benchmark_results/rs_hnsw_sift128.full_k100.json` locally and confirm the row reports:
- `adaptive_k == 0.0`
- `requested_ef_search == 138`
- `effective_ef_search == 138`

### Task 3: Refresh Derived Fairness Artifacts

**Files:**
- Modify: `benchmark_results/baseline_p3_001_same_schema_hnsw_hdf5.json`
- Modify: `benchmark_results/hnsw_fairness_gate.json`

- [ ] **Step 1: Rebuild the same-schema summary from the new Rust row**

Update the Rust row, derived qps ratio, and notes in `benchmark_results/baseline_p3_001_same_schema_hnsw_hdf5.json` so the same-schema artifact points at the fresh authoritative fair-lane Rust evidence.

- [ ] **Step 2: Rebuild the fairness gate summary**

Update `benchmark_results/hnsw_fairness_gate.json` so:
- requested `ef` alignment still passes
- effective `ef` alignment now passes
- datatype alignment still fails
- query-dispatch alignment still fails
- `fair_for_leadership_claim` remains `false`

- [ ] **Step 3: Run the tightened fairness contract**

Run: `python3 -m unittest tests/test_hnsw_fairness_gate.py tests/test_baseline_methodology_lock.py`
Expected: `OK`

### Task 4: Durable Closure

**Files:**
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`

- [ ] **Step 1: Record the authority rerun**

Add a new session entry in `task-progress.md` with:
- focus
- commands run
- retained remote log path
- result (`effective_ef` blocker cleared, datatype/dispatch still open)

- [ ] **Step 2: Update release notes**

Add one `Unreleased` note describing the fair-lane authority rerun and what remains blocked.

- [ ] **Step 3: Run final verification**

Run:
- `python3 scripts/validate_features.py feature-list.json`
- `git status --short`

Expected:
- validator stays `66/66 passing`
- worktree contains only the files from this authority slice until commit
