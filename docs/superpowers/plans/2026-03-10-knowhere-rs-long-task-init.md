# knowhere-rs Long-Task Workflow Initialization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Initialize a durable long-task workflow for the non-GPU `knowhere-rs` program, using the existing remote x86 machine as the sole authority environment.

**Architecture:** The workflow is centered on durable project files in the repository root. `feature-list.json` is the task truth source, `task-progress.md` is the session log, and `init.sh` bootstraps the remote authority environment before any feature work starts.

**Tech Stack:** Markdown, JSON, Bash, PowerShell, Python 3, existing `scripts/remote/*` automation

---

## Chunk 1: Durable Spec and Plan

### Task 1: Write the durable design spec

**Files:**
- Create: `docs/superpowers/specs/2026-03-10-knowhere-rs-long-task-design.md`

- [ ] **Step 1: Write the design spec**

Capture the goal, scope, remote-authority rule, task taxonomy, durable files, and completion definition for the long-task workflow.

- [ ] **Step 2: Verify the spec exists**

Run: `test -f docs/superpowers/specs/2026-03-10-knowhere-rs-long-task-design.md`
Expected: success

### Task 2: Write the initialization plan

**Files:**
- Create: `docs/superpowers/plans/2026-03-10-knowhere-rs-long-task-init.md`

- [ ] **Step 1: Write the initialization plan**

Describe how to create the durable artifacts, customize them for remote x86 authority, validate them, and record Session 0.

- [ ] **Step 2: Verify the plan exists**

Run: `test -f docs/superpowers/plans/2026-03-10-knowhere-rs-long-task-init.md`
Expected: success

## Chunk 2: Durable Workflow Files

### Task 3: Populate the feature inventory

**Files:**
- Modify: `feature-list.json`
- Create: `scripts/validate_features.py`

- [ ] **Step 1: Copy the validator**

Run: `cp /Users/ryan/.openclaw/skills/long-task-codex/scripts/validate_features.py scripts/validate_features.py`
Expected: file copied

- [ ] **Step 2: Replace the placeholder inventory**

Add baseline, HNSW, IVF-PQ, DiskANN, cross-cutting production, and final-acceptance features with concrete remote-x86 verification steps and dependency edges.

- [ ] **Step 3: Validate the feature inventory**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID - ... features`

### Task 4: Customize worker instructions and session state

**Files:**
- Modify: `long-task-guide.md`
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`

- [ ] **Step 1: Make the worker guide remote-first**

State that the existing remote x86 machine is the sole authority environment and that local results do not justify `passing`.

- [ ] **Step 2: Seed Session 0**

Update `task-progress.md` with the requirement sources, design/plan paths, current focus, and next feature.

- [ ] **Step 3: Record initialization in release notes**

Note the new durable workflow artifacts and remote-authority rule.

## Chunk 3: Bootstrap Entry Point

### Task 5: Replace the generic bootstrap stub with a remote-x86 bootstrap

**Files:**
- Modify: `init.sh`
- Modify: `init.ps1`

- [ ] **Step 1: Update `init.sh`**

Make it load `scripts/remote/common.sh`, print the resolved remote configuration, sync the repo to the remote authority machine, and probe the remote Rust toolchain.

- [ ] **Step 2: Update `init.ps1`**

Document that the supported bootstrap is `bash init.sh` via a POSIX shell.

- [ ] **Step 3: Run bootstrap smoke**

Run: `bash init.sh`
Expected: remote config printed, sync succeeds, remote cargo/rustc versions print

## Chunk 4: Persistence and Handoff

### Task 6: Final validation and handoff

**Files:**
- Modify: `task-progress.md`
- Modify: `RELEASE_NOTES.md`

- [ ] **Step 1: Re-run feature validation**

Run: `python3 scripts/validate_features.py feature-list.json`
Expected: `VALID - ... features`

- [ ] **Step 2: Inspect new durable files**

Run: `git status --short`
Expected: new/modified workflow files visible and scoped to initialization work

- [ ] **Step 3: Commit the initialization**

```bash
git add AGENTS.md feature-list.json long-task-guide.md task-progress.md RELEASE_NOTES.md init.sh init.ps1 scripts/validate_features.py docs/superpowers/specs/2026-03-10-knowhere-rs-long-task-design.md docs/superpowers/plans/2026-03-10-knowhere-rs-long-task-init.md
git commit -m "docs(long-task): initialize remote-x86 workflow"
```
