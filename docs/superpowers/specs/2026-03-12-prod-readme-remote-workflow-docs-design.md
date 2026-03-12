# Prod Readme Remote Workflow Docs Design

## Goal

Close `prod-readme-remote-workflow-docs` by aligning the operator-facing entry docs with the repository's actual remote-first workflow and current benchmark truth.

## Problem

The current entry docs are materially stale:

- `README.md` still presents the project as a local-only Rust library with generic `cargo build` / `cargo test` guidance and an obsolete `34 tests passed` claim
- `AGENTS.md` lists local build/test commands but does not clearly state that remote x86 is the only acceptance authority
- `docs/FFI_CAPABILITY_MATRIX.md` still ends with generic local-only validation instructions even though production contract closure now depends on remote replay

That stale guidance is dangerous because future sessions or human operators can follow the wrong workflow and create false-green conclusions.

## Options

### Option 1: Rewrite the entry docs around the actual workflow

Update the small set of docs that new operators or sessions read first:

- `README.md`
- `AGENTS.md`
- `docs/FFI_CAPABILITY_MATRIX.md`

Pros:

- directly fixes the stale operator path
- keeps scope tight
- matches the feature title and recorded verification steps

Cons:

- does not try to normalize every historical design note in `docs/`

### Option 2: Sweep every markdown file in the repository

Pros:

- could eliminate more stale language in one pass

Cons:

- scope explosion
- many historical/per-feature design docs are intentionally archival and should not be rewritten to sound current

### Option 3: Update only `README.md`

Pros:

- smallest change

Cons:

- leaves `AGENTS.md` and the FFI operator matrix inconsistent with the new workflow

## Decision

Take Option 1.

The feature is about operator-facing workflow docs, not a full archive rewrite.

## Scope

This feature will:

1. rewrite `README.md` into a remote-first project/operator overview
2. update `AGENTS.md` so build/test guidance distinguishes local prefilters from remote acceptance
3. update `docs/FFI_CAPABILITY_MATRIX.md` so contract validation guidance points to remote replay
4. verify with:
   - `bash init.sh`
   - `python3 scripts/validate_features.py feature-list.json`
   - `bash scripts/remote/build.sh --no-all-targets`

## Non-Goals

This feature will not:

- rewrite historical benchmark deep-dive docs
- reopen any family verdict
- change runtime behavior or benchmark artifacts

## Design

### 1. README as the remote-first landing page

`README.md` should answer four questions immediately:

1. what this repository is
2. what the current benchmark/verdict truth is
3. how operators actually work with the remote authority machine
4. where the durable state lives

### 2. AGENTS guidance should stop implying local-only acceptance

`AGENTS.md` should still keep the local developer commands, but it must explicitly say:

- remote x86 is the acceptance authority
- local commands are prefilters
- `feature-list.json` verification steps are the gate for long-task work

### 3. FFI capability doc should point to the right verification path

The capability matrix is an operator-facing contract summary, so its validation section should:

- keep the quick local smoke examples
- add the remote authority replay commands
- state clearly that the matrix is a contract view, not a performance verdict

## Testing Strategy

1. edit the operator-facing docs
2. run `bash init.sh`
3. run `bash scripts/remote/build.sh --no-all-targets`
4. run `python3 scripts/validate_features.py feature-list.json`

## Risks

### Risk: mixing operator guidance with archival benchmark notes

Mitigation:

Only rewrite the landing/guide docs that should always read as current.

### Risk: over-promising project readiness in README

Mitigation:

State the current verdicts plainly: HNSW is usable but not leading, IVF-PQ is no-go, DiskANN is constrained, and final leadership proof is not met.
