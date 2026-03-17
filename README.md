# knowhere-rs

`knowhere-rs` 是一个面向 non-GPU 向量检索场景的 Rust 实现，当前按 remote-first workflow 维护 authority 证据链与生产验收状态。

## Current Status

- Authority environment: 现有 remote x86 机器是唯一权威执行面；本地 `cargo` 结果只用于快速预筛
- HNSW: `leading`（near-equal-recall authority lane）
- IVF-PQ: `no-go`
- DiskANN: `constrained`
- Final performance leadership criterion: `criterion_met=true`
- Production governance: remote `fmt/clippy/build`、cross-cutting `ffi/serialize/bench_json_export`、以及 operator docs gates 已关闭
- Project-level verdict: `final-production-acceptance` 已归档为 `production_accepted=true`

当前 benchmark / verdict truth 见：

- `benchmark_results/final_core_path_classification.json`
- `benchmark_results/final_performance_leadership_proof.json`
- `benchmark_results/final_production_acceptance.json`
- `docs/parity/knowhere-rs-vs-native-2026-03-17.md`
- `TASK_QUEUE.md`
- `GAP_ANALYSIS.md`
- `docs/PARITY_AUDIT.md`

## Remote-First Workflow

1. 初始化 authority workspace

```bash
bash init.sh
```

2. 读取 durable state

- `long-task-guide.md`
- `task-progress.md`
- `feature-list.json`

3. 本地只做预筛

```bash
cargo test --lib ffi -- --nocapture
cargo test --lib serialize -- --nocapture
```

4. 在 remote x86 上执行真正的验收命令

```bash
bash scripts/remote/test.sh --command "cargo test --lib -q"
bash scripts/remote/build.sh --no-all-targets
```

5. 更新 durable state 并校验

```bash
python3 scripts/validate_features.py feature-list.json
```

需要隔离远端 cache / logs 时，使用 feature-specific 目录：

```bash
KNOWHERE_RS_REMOTE_TARGET_DIR=/data/work/knowhere-rs-target-<feature> \
KNOWHERE_RS_REMOTE_LOG_DIR=/data/work/knowhere-rs-logs-<feature> \
bash scripts/remote/test.sh --command "cargo test --lib ffi -- --nocapture"
```

## Common Commands

本地预筛：

```bash
cargo build --verbose
cargo test --lib --verbose
cargo test --tests --verbose
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

Authority replay：

```bash
bash init.sh
bash scripts/remote/test.sh --command "cargo test --lib -q"
bash scripts/remote/build.sh --no-all-targets
```

## Durable State

未来 session 或人工操作默认从这些文件恢复上下文：

- `task-progress.md`: 当前 focus、最近 session、fresh verification
- `feature-list.json`: feature inventory、依赖、verification steps
- `TASK_QUEUE.md`: 当前大任务面板
- `GAP_ANALYSIS.md`: 项目级缺口与完成定义
- `docs/PARITY_AUDIT.md`: 审计轨迹与阶段性 verdict
- `docs/FFI_CAPABILITY_MATRIX.md`: FFI contract 视图

## Repository Layout

```text
src/
  faiss/         core index implementations
  ffi/           FFI-facing helpers
  benchmark/     benchmark artifact/report schema helpers
tests/           integration and regression tests
benches/         Criterion microbenchmarks
scripts/remote/  remote authority bootstrap/test/build wrappers
docs/            design notes, parity audit, and operator docs
```

## Notes

- 不要把本地 benchmark 或本地 test 结果当成最终 acceptance evidence
- 不要在没有 remote verification 的情况下把 feature 标成 `passing`
- 当前仓库已经形成 durable multi-session workflow；开始工作前优先读取 `long-task-guide.md`
