# Prod All Targets Clippy Fmt Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remote formatting, clippy, and all-targets build gate without weakening verification.

**Architecture:** Treat the feature as three linked closures. First, add a small regression that keeps `build.rs` and `Cargo.toml` aligned for the `faiss-cxx` build path. Second, make the repo rustfmt-clean. Third, fix only the warning/error surface that currently blocks `cargo clippy --all-targets --all-features -- -D warnings`, then verify the unchanged remote commands.

**Tech Stack:** Rust 2021, `cargo fmt`, `cargo clippy`, Python `unittest`, remote authority wrappers in `scripts/remote/`, durable long-task state files

---

## Chunk 1: Lock The Build-Dependency Hygiene

### Task 1: Add a failing regression for the `faiss-cxx` build path manifest drift

**Files:**
- Create: `tests/test_prod_gate_hygiene.py`
- Modify: `Cargo.toml`

- [ ] **Step 1: Write the failing hygiene unittest**

Add `tests/test_prod_gate_hygiene.py` with a test that reads `build.rs` and `Cargo.toml` and asserts:

- `build.rs` contains the `cfg(feature = "faiss-cxx")` gate
- `[build-dependencies]` contains `cxx-build`

- [ ] **Step 2: Run the unittest to verify RED**

Run: `python3 -m unittest tests/test_prod_gate_hygiene.py`
Expected: FAIL because `Cargo.toml` does not yet declare `cxx-build`.

- [ ] **Step 3: Add the minimal manifest fix**

Add `cxx-build` to `[build-dependencies]` in `Cargo.toml`.

- [ ] **Step 4: Run the unittest to verify GREEN**

Run: `python3 -m unittest tests/test_prod_gate_hygiene.py`
Expected: PASS

## Chunk 2: Close The Formatting Gate

### Task 2: Bring the repository to rustfmt-clean state

**Files:**
- Modify: Rust files touched by `cargo fmt --all`

- [ ] **Step 1: Run the formatting gate to verify RED**

Run: `cargo fmt --all -- --check`
Expected: FAIL with formatting diffs.

- [ ] **Step 2: Apply the formatter**

Run: `cargo fmt --all`

- [ ] **Step 3: Re-run the formatting gate to verify GREEN**

Run: `cargo fmt --all -- --check`
Expected: PASS

## Chunk 3: Close The Clippy Gate

### Task 3: Fix the current authority clippy surface

**Files:**
- Modify: files reported by `cargo clippy --all-targets --all-features -- -D warnings`
- Test: `tests/test_prod_gate_hygiene.py`

- [ ] **Step 1: Run clippy to verify RED**

Run: `cargo clippy --all-targets --all-features -- -D warnings`
Expected: FAIL, first on the missing build dependency before the manifest fix, then on the actual warning surface.

- [ ] **Step 2: Make the smallest warning fixes**

For each reported blocking warning:

- remove unused imports/variables when they are truly dead
- remove unnecessary `mut` / `unsafe`
- adjust test attributes or `cfg` scopes when benchmark-only helpers are leaking into all-targets gates
- use targeted `#[allow(...)]` only when the warning is intentional and documented by code structure

- [ ] **Step 3: Re-run clippy to verify GREEN**

Run: `cargo clippy --all-targets --all-features -- -D warnings`
Expected: PASS

## Chunk 4: Verify On The Authority Machine And Persist State

### Task 4: Run the recorded remote gate and sync durable state

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Refresh remote workspace**

Run: `bash init.sh`
Expected: PASS

- [ ] **Step 2: Run the recorded remote verification commands**

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-prod-clippy-fmt KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-prod-clippy-fmt bash scripts/remote/test.sh --command "cargo fmt --all -- --check"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-prod-clippy-fmt KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-prod-clippy-fmt bash scripts/remote/test.sh --command "cargo clippy --all-targets --all-features -- -D warnings"`
Expected: `test=ok`

Run: `KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-prod-clippy-fmt KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-prod-clippy-fmt bash scripts/remote/build.sh`
Expected: `build=ok`

- [ ] **Step 3: Update durable state and validate**

Mark `prod-all-targets-clippy-fmt` as `passing`, record the remote logs and the gate-closure scope in the durable files, then run:

`python3 scripts/validate_features.py feature-list.json`

Expected: `VALID`
