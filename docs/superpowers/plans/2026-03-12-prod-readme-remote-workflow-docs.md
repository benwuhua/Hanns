# Prod Readme Remote Workflow Docs Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans or superpowers:subagent-driven-development before changing files from this plan.

**Goal:** Replace stale local-only operator guidance with the repository's actual remote-first workflow and current benchmark truth.

**Architecture:** Update the three entry/operator docs that shape future session behavior: `README.md`, `AGENTS.md`, and `docs/FFI_CAPABILITY_MATRIX.md`. Keep the scope limited to operator guidance. Do not rewrite archival design/benchmark notes that are intentionally historical.

**Tech Stack:** Markdown docs, remote authority wrappers, durable workflow state files

---

## Chunk 1: Rewrite The Entry Docs

### Task 1: Make `README.md` match the actual workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Remove stale local-only claims**

Delete obsolete statements such as generic `cargo test` acceptance and static passed-test counts.

- [ ] **Step 2: Add the current project truth**

Document:

- remote x86 as the only authority
- HNSW=`functional-but-not-leading`
- IVF-PQ=`no-go`
- DiskANN=`constrained`
- `criterion_met=false` for final leadership proof

- [ ] **Step 3: Add the operator workflow**

Document:

- `bash init.sh`
- durable-state files to read first
- local smoke vs remote acceptance commands

## Chunk 2: Align Supporting Operator Docs

### Task 2: Update `AGENTS.md` and `docs/FFI_CAPABILITY_MATRIX.md`

**Files:**
- Modify: `AGENTS.md`
- Modify: `docs/FFI_CAPABILITY_MATRIX.md`

- [ ] **Step 1: Add a remote-authority note to `AGENTS.md`**

Make it explicit that local cargo commands are prefilters and remote verification is the acceptance gate.

- [ ] **Step 2: Update the FFI matrix validation guidance**

Keep local smoke examples, but add `bash init.sh` and remote replay examples.

## Chunk 3: Verify And Sync Durable State

### Task 3: Close the feature with remote-safe verification and durable updates

**Files:**
- Modify: `feature-list.json`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`
- Modify: `TASK_QUEUE.md`
- Modify: `GAP_ANALYSIS.md`
- Modify: `DEV_ROADMAP.md`
- Modify: `docs/PARITY_AUDIT.md`

- [ ] **Step 1: Run the recorded verification steps**

Run:

- `bash init.sh`
- `python3 scripts/validate_features.py feature-list.json`
- `bash scripts/remote/build.sh --no-all-targets`

Expected: all pass

- [ ] **Step 2: Mark the feature passing**

Update durable state so `prod-readme-remote-workflow-docs` becomes `passing`.

- [ ] **Step 3: Advance the queue**

Record the next ready feature chosen from `feature-list.json` after this docs gate closes.
