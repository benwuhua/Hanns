# Fast-Lane Authority Workflow Design

Date: 2026-03-13
Status: Proposed
Scope: long-task workflow optimization for faster single-round performance verdicts

## Problem

The current reopen workflow is too expensive for narrow performance hypotheses.

A single HNSW reopen round currently tends to include all of the following:

- activation and baseline artifact creation
- local audit/profile implementation
- authority same-schema rerun
- authority replay of audit/default contracts
- durable closure updates across `feature-list.json`, `task-progress.md`, `RELEASE_NOTES.md`, and `docs/PARITY_AUDIT.md`

This gives good traceability, but it also means a weak hypothesis pays nearly the full durable-state cost before the authority lane can reject it. That slows down the total elapsed time from idea to verdict.

The specific bottlenecks are:

- too many steps before the first authority decision
- too much durable-document churn for hypotheses that later hard-stop
- repeated context switching between code, audit artifacts, authority commands, and workflow bookkeeping
- `feature-list.json` being used for both real authority-tracked work and very early local exploration

## Goal

Reduce total wall-clock time from hypothesis to trusted verdict without weakening the core project rule:

- remote x86 remains the only authoritative execution surface for performance and stop-go claims

The workflow should make it cheaper to reject weak ideas early, while preserving the existing durable evidence chain for hypotheses that are worth taking through authority.

## Non-Goals

- replacing remote authority verification with local performance claims
- removing durable artifacts for accepted authority results
- changing benchmark methodology, recall gates, or same-schema comparison policy
- changing the existing historical evidence already archived in the repository

## Recommended Approach

Adopt a two-tier workflow:

1. `screen` lane for fast local hypothesis triage
2. `tracked authority round` only after a hypothesis passes the screen gate

This replaces the current pattern of immediately opening a full tracked reopen round for every plausible idea.

## Alternative Approaches Considered

### Option A: Keep the current workflow and only optimize scripts

Examples:

- parallelize more local commands
- reduce duplicate local checks
- shave seconds from sync/test wrappers

Pros:

- lowest workflow disruption

Cons:

- does not solve the real cost center, which is full-round durable overhead before authority verdict
- weak hypotheses still consume activation, audit, and closure work

Verdict:

- not recommended as the primary fix

### Option B: Compress current tracked rounds into fewer durable phases

Examples:

- remove separate activation closure
- combine audit and summary closure

Pros:

- smaller change to operator habits

Cons:

- still opens tracked rounds too early
- still makes failed ideas expensive

Verdict:

- acceptable fallback, but weaker than a real screen gate

### Option C: Add a fast local screening lane, then promote only strong hypotheses

Pros:

- attacks the main cost directly
- keeps authority as the only truth source
- reduces durable churn for hard-stop ideas

Cons:

- requires explicit distinction between exploratory work and authority-tracked work

Verdict:

- recommended

## Proposed Workflow

### Phase 1: Screen

Purpose:

- answer one question quickly: is this hypothesis strong enough to spend authority time on?

Inputs:

- one narrowly scoped hypothesis
- one expected mechanism
- one local acceptance threshold

Allowed work:

- focused red/green regression
- minimal code change
- one small local benchmark, audit, or profile
- brief temporary notes in `task-progress.md`

Explicitly not required:

- `feature-list.json` reopening
- baseline artifact creation
- authority summary artifact
- `RELEASE_NOTES.md` update
- `docs/PARITY_AUDIT.md` update

Outputs:

- `screen_result = promote | reject | needs_more_local`
- one short temporary record in `task-progress.md`

Promotion criteria:

- the hypothesis is still narrow and attributable
- the local signal crosses a predeclared threshold
- there is no obvious regression against current constraints

Rejection criteria:

- no meaningful local signal
- signal depends on broad or ambiguous changes
- the hypothesis conflicts with prior authority evidence

Iteration cap:

- a single hypothesis should not stay in `screen` indefinitely
- default cap: at most 1 follow-up local refinement after the initial screen result of `needs_more_local`

### Phase 2: Authority

Purpose:

- obtain the first trusted verdict with the smallest authoritative command set

Default commands:

- `bash init.sh`
- one authority Rust same-schema rerun
- one fresh native capture
- one remote replay only if needed for the new contract or artifact

Outputs:

- one authority summary artifact
- one verdict: `continue | soft_stop | hard_stop`

Design rule:

- authority should not wait on full durable reopening
- if a hypothesis passed screen, authority becomes the next immediate gate

### Phase 3: Durable Closure

Purpose:

- update long-lived repository state only after authority has produced a trusted verdict

Required updates:

- `feature-list.json`
- `task-progress.md`
- `RELEASE_NOTES.md`
- `docs/PARITY_AUDIT.md`

Design rule:

- durable state is written once per successful authority-backed round, not continuously throughout early exploration

## Tracked vs Untracked Work

### Untracked Screen Work

Untracked does not mean undocumented. It means:

- not yet added as a formal `feature-list.json` feature
- not yet treated as a durable authority-backed round

Screen work is still recorded in `task-progress.md`, but as temporary exploration rather than as a passing/failing tracked feature.

### Tracked Authority Rounds

A hypothesis becomes tracked only after promotion from screen.

At that point, the workflow opens a formal round with:

- explicit baseline linkage
- explicit authority summary artifact
- explicit durable closure

This keeps `feature-list.json` focused on true authority-relevant work instead of early local exploration.

## Policy Changes

### 1. Hard-stop families cannot reopen directly

If a family or sub-line already ended in `hard_stop`, a new hypothesis must first pass `screen`.

Rationale:

- prevents repeated expensive reopen cycles driven only by intuition

### 2. `feature-list.json` tracks authority-grade work only

Local screening experiments should not immediately become tracked features.

Rationale:

- reduces inventory churn
- keeps the validator and durable workflow focused on evidence-bearing work

### 3. Every screen must declare an expected mechanism and threshold up front

Examples:

- “remove hot-loop profiling overhead; expect at least X% local qps gain”
- “improve locality; expect at least Y% local distance-stage reduction”

Rationale:

- avoids drifting into vague exploratory edits

## Expected Benefits

For weak hypotheses:

- much faster rejection
- no full round activation cost
- no audit artifact churn before authority evidence exists

For strong hypotheses:

- less time spent before the first authority verdict
- fewer durable edits while implementation is still fluid

For repository hygiene:

- cleaner `feature-list.json`
- fewer partial reopen states
- less document churn per idea

## Risks

### Risk: useful local exploration becomes too informal

Mitigation:

- require a brief `task-progress.md` note for every screen result

### Risk: operators may skip durable updates after authority

Mitigation:

- authority completion must still end with the current durable closure requirements

### Risk: local screen thresholds become arbitrary

Mitigation:

- every screen must declare expected mechanism and threshold before edits start

### Risk: too many ideas stay in endless local refinement

Mitigation:

- default cap of one follow-up screen attempt before reject or promote

## Rollout Plan

1. Update workflow docs to define `screen`, `authority`, and `durable` phases.
2. Clarify that `feature-list.json` is for authority-tracked work, not pre-screen exploration.
3. Add validator rules only if needed after the documentation change is stable.
4. Apply the new workflow first to future HNSW reopen attempts, then expand to other performance families if it works well.

## Success Criteria

The workflow is successful if future narrow performance ideas:

- reach their first authority verdict with fewer prerequisite steps
- produce less durable churn when they hard-stop
- keep the same authority-only standard for benchmark and verdict claims
- leave `feature-list.json` cleaner and more stable than the current reopen-heavy pattern

## Open Questions

- whether temporary `screen` notes should follow a stricter template inside `task-progress.md`
- whether the validator should eventually detect invalid “tracked before screen” patterns for hard-stop families
- whether some families besides HNSW should continue to use the older fully tracked workflow by default
