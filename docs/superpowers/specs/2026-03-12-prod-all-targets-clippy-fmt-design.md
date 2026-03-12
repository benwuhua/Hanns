# Prod All Targets Clippy Fmt Design

## Goal

Close `prod-all-targets-clippy-fmt` by making the repository honestly pass the remote authority formatting, clippy, and all-targets build gates without weakening any verification step.

## Problem

The next production gate is no longer blocked by benchmark ambiguity. It is blocked by toolchain hygiene:

- `cargo fmt --all -- --check` currently fails with widespread formatting drift
- `cargo clippy --all-targets --all-features -- -D warnings` currently fails before linting because `build.rs` references `cxx_build` behind `faiss-cxx`, but `Cargo.toml` does not declare the corresponding build dependency
- a remote `scripts/remote/build.sh` probe appears to compile through, but it emits a large warning surface that will become hard failures once clippy is enforced with `-D warnings`

This means the project still relies on local tolerance for style/lint drift. That is exactly what this feature is supposed to eliminate.

## Options

### Option 1: Honest gate closure

Add the missing build-dependency hygiene, run repo-wide rustfmt, then fix only the warning classes that actually block the current authority `clippy --all-targets --all-features -D warnings` lane.

Pros:

- preserves the existing verification steps unchanged
- leaves a replayable gate closure instead of a documentation-only claim
- keeps scope tied to actual authority failures rather than speculative cleanup

Cons:

- likely touches multiple files because rustfmt and clippy warnings are cross-cutting

### Option 2: Partial blocker cleanup

Fix `cxx-build` and maybe run rustfmt, but stop before the full clippy lane is green.

Pros:

- smaller change set

Cons:

- does not close the feature
- leaves the production acceptance dependency chain blocked

### Option 3: Suppress warnings broadly

Use crate-level or wide `allow(...)` annotations to silence the authority clippy lane.

Pros:

- fastest path to a green command

Cons:

- weakens the production gate
- hides real hygiene debt instead of resolving it

## Decision

Take Option 1.

The repository already has a trustworthy remote-x86 harness. The correct move now is to make the code and manifest state satisfy the existing gates, not to redefine the gates.

## Scope

This feature will:

1. add a lightweight regression that keeps `build.rs` and `Cargo.toml` consistent for the `faiss-cxx` build path
2. make the repository rustfmt-clean for `cargo fmt --all -- --check`
3. make the repository clippy-clean for `cargo clippy --all-targets --all-features -- -D warnings`
4. verify the unchanged remote authority steps:
   - `bash scripts/remote/test.sh --command "cargo fmt --all -- --check"`
   - `bash scripts/remote/test.sh --command "cargo clippy --all-targets --all-features -- -D warnings"`
   - `bash scripts/remote/build.sh`

## Non-Goals

This feature will not:

- weaken or remove any verification step
- change benchmark verdicts or acceptance policy
- reopen family-level performance work
- do unrelated refactoring outside the warnings needed for the authority clippy lane

## Design

### 1. Build-dependency hygiene lock

Add a small Python unittest, following the existing gate-hygiene pattern, that asserts:

- `build.rs` still gates the CXX bridge behind `feature = "faiss-cxx"`
- `Cargo.toml` declares a `cxx-build` entry in `[build-dependencies]`

This does two things:

- provides a true TDD red/green entrypoint for the missing build dependency
- prevents future regressions where `clippy --all-features` breaks before linting due to manifest drift

### 2. Formatting closure

Use `cargo fmt --all` to bring the working tree to the formatter’s canonical style, then rerun `cargo fmt --all -- --check`.

The feature is explicitly about remote fmt closure, so broad formatting churn is in scope as long as it stays mechanical and does not change behavior.

### 3. Clippy closure

After the build-dependency fix and formatting pass, rerun `cargo clippy --all-targets --all-features -- -D warnings` and fix only the currently reported warning/error surface.

Expected warning classes based on the current probes:

- unused imports
- unnecessary `mut`
- unnecessary `unsafe`
- unused variables such as `remainder`
- dead code in benchmark helper paths that should be gated or explicitly allowed only where justified
- outdated test attributes such as `#[ignore]` applied to macro calls

The rule is narrowness:

- prefer deleting or renaming unused items
- prefer feature/`cfg` scoping for long-benchmark helpers
- use `#[allow(...)]` only when the warning is intentional and the code should remain visible to readers

### 4. Authority verification

The feature is only complete when the remote x86 commands pass unchanged.

`scripts/remote/build.sh` remains part of the verification surface even if it is already close to green, because this feature is the gate that says production acceptance no longer depends on local-only lint/style checks.

## Testing Strategy

1. add the failing hygiene unittest for the missing `cxx-build` relationship
2. run it red
3. fix the manifest/build-script relationship and run it green
4. run `cargo fmt --all -- --check` red, then `cargo fmt --all`, then green
5. run `cargo clippy --all-targets --all-features -- -D warnings` red and iteratively green
6. run the recorded remote verification commands
7. sync durable workflow state

## Risks

### Risk: rustfmt introduces wide churn

Mitigation:

Keep the pass mechanical and avoid mixing semantic edits into the formatting-only phase.

### Risk: clippy cleanup expands beyond one feature

Mitigation:

Only fix warnings currently reported by the authority gate. Do not proactively “clean the whole repo” beyond what the gate requires.

### Risk: build-dependency fix changes optional build behavior

Mitigation:

The new dependency only makes the build script compile when `faiss-cxx` is enabled. The existing `cfg(feature = "faiss-cxx")` block still controls whether the bridge is actually invoked.
