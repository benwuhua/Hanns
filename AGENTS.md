# Repository Guidelines

## Project Structure & Module Organization
- Core Rust library code lives in `src/`, with major domains split into modules like `src/faiss/`, `src/clustering/`, `src/quantization/`, `src/dataset/`, and `src/ffi/`.
- CLI entry point: `src/bin/cli.rs` (`knowhere-cli`).
- Integration and regression coverage is primarily in `tests/` (many scenario-style files such as `test_*.rs`, `bench_*.rs`, `debug_*.rs`).
- Microbenchmarks using Criterion live in `benches/`.
- Helper scripts and dataset utilities live in `scripts/` (for example `scripts/download_sift1m.sh`).
- C headers are under `include/`; examples are in `examples/`; design notes and reports are in `docs/` and root `*.md` files.

## Build, Test, and Development Commands
- `cargo build --verbose`: debug build of the library and binaries.
- `cargo build --release --verbose`: optimized build for benchmarks/perf checks.
- `cargo test --lib --verbose`: unit tests in library modules.
- `cargo test --tests --verbose`: integration tests in `tests/`.
- `cargo clippy --all-targets --all-features -- -D warnings`: lint gate used by CI.
- `cargo fmt --all -- --check`: formatting gate used by CI.
- `cargo test --release --test perf_test -- --nocapture --test-threads=1`: optional perf smoke test used on main branch CI.
- `./build.sh release`: project wrapper for release build.

## Coding Style & Naming Conventions
- Rust edition is 2021 (`Cargo.toml`); keep code `rustfmt`-clean and clippy-clean.
- Use 4-space indentation and idiomatic Rust naming: `snake_case` for functions/files, `CamelCase` for types/traits, `SCREAMING_SNAKE_CASE` for constants.
- Prefer small, focused modules and keep FFI-facing changes mirrored in `include/` headers when needed.

## Testing Guidelines
- Put fast unit tests next to code with `#[cfg(test)] mod tests`.
- Put cross-module behavior tests in `tests/` and name files by feature, e.g. `test_diskann_aisaq.rs`.
- For benchmarks, use `benches/*.rs` with Criterion (`cargo bench`).
- Run at least `fmt`, `clippy`, `cargo test --lib`, and `cargo test --tests` before opening a PR.

## Commit & Pull Request Guidelines
- Follow Conventional Commit style seen in history: `feat(scope): ...`, `fix(scope): ...`, or `feat: ...`.
- Keep commits focused by subsystem (example: `feat(idx-24): SPARSE_WAND ...`).
- PRs should include: purpose, key changes, test commands/results, and linked task/issue IDs (`IDX-*`, `BENCH-*`, `FFI-*`) when applicable.
- If changes affect performance or FFI behavior, attach benchmark notes and compatibility impact in the PR description.
