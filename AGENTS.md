# Repository Guidelines

## Project Structure & Module Organization
- Core Rust library code lives in `src/`, with major domains split into modules like `src/faiss/`, `src/clustering/`, `src/quantization/`, `src/dataset/`, and `src/ffi/`.
- CLI entry point: `src/bin/cli.rs` (`knowhere-cli`).
- Integration and regression coverage is primarily in `tests/` (many scenario-style files such as `test_*.rs`, `bench_*.rs`, `debug_*.rs`).
- Microbenchmarks using Criterion live in `benches/`.
- Helper scripts and dataset utilities live in `scripts/` (for example `scripts/download_sift1m.sh`).
- C headers are under `include/`; examples are in `examples/`; design notes and reports are in `docs/` and root `*.md` files.

## Authority Workflow
- The existing remote x86 machine is the only authoritative execution surface for long-task production acceptance and benchmark/verdict claims.
- Local `cargo` commands are for quick iteration and smoke checks only.
- Narrow performance ideas should start in a local `screen` phase and only become tracked work after `screen_result=promote`.
- Before making production benchmark/verdict claims, run the relevant remote x86 verification via `scripts/remote/` helpers and archive the resulting evidence.

## Reasoning Principles
- Default to first-principles reasoning: reduce each problem to irreducible facts, explicit constraints, and the actual acceptance target before choosing an approach.
- Distinguish hard requirements from legacy implementation details, conventions, or accidental complexity; do not preserve historical structure unless it is required.
- Prefer solutions derived from root causes and minimal necessary interfaces over cargo-culted parity work or surface-level patching.
- When tradeoffs exist, explain the constraint chain that leads to the decision so the reasoning is inspectable and falsifiable.

## Build, Test, and Development Commands
- `cargo build --verbose`: debug build of the library and binaries.
- `cargo build --release --verbose`: optimized build for benchmarks/perf checks.
- `cargo test --lib --verbose`: unit tests in library modules.
- `cargo test --tests --verbose`: integration tests in `tests/`.
- `cargo clippy --all-targets --all-features -- -D warnings`: lint gate used by CI.
- `cargo fmt --all -- --check`: formatting gate used by CI.
- `cargo test --release --test perf_test -- --nocapture --test-threads=1`: optional perf smoke test used on main branch CI.
- `bash scripts/build.sh release`: project wrapper for release build plus tests.
- `bash scripts/remote/sync.sh --mode rsync`: sync the current workspace to the remote x86 authority machine when remote verification is needed.
- `bash scripts/remote/test.sh --command "<cargo command>"`: authoritative remote test execution wrapper.
- `bash scripts/remote/build.sh --no-all-targets`: authoritative remote build smoke when the feature only needs the production build lane.

## Coding Style & Naming Conventions
- Rust edition is 2021 (`Cargo.toml`); keep code `rustfmt`-clean and clippy-clean.
- Use 4-space indentation and idiomatic Rust naming: `snake_case` for functions/files, `CamelCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants.
- Prefer small, focused modules and keep FFI-facing changes mirrored in `include/` headers when needed.

## Testing Guidelines
- Put fast unit tests next to code with `#[cfg(test)] mod tests`.
- Put cross-module behavior tests in `tests/` and name files by feature, e.g. `tests/bench_diskann_1m.rs`.
- For benchmarks, use `benches/*.rs` with Criterion (`cargo bench`).
- Run at least `fmt`, `clippy`, `cargo test --lib`, and `cargo test --tests` before opening a PR.
- For production benchmark/verdict work, treat local commands as prefilters only; archived remote-x86 evidence is the acceptance gate.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history: `feat(scope): ...`, `fix(scope): ...`, or `feat: ...`.
- Keep commits focused by subsystem (example: `feat(idx-24): SPARSE_WAND ...`).
- PRs should include: purpose, key changes, test commands/results, and linked task/issue IDs (`IDX-*`, `BENCH-*`, `FFI-*`) when applicable.
- If changes affect performance or FFI behavior, attach benchmark notes and compatibility impact in the PR description.
