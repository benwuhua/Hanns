# Gate Profile Runner

统一门禁入口：`scripts/gate_profile_runner.sh`

## Profile 映射

- `default_regression` -> `cargo test -q`
- `full_regression` ->
  - `cargo test --lib -q`
  - `cargo test --tests -q`
  - `cargo test --doc -q`
  - `python3 -m unittest tests/test_validate_features.py`
  - `python3 -m unittest tests/test_baseline_methodology_lock.py`
  - `python3 -m unittest tests/test_parity_report_lock.py`
  - `python3 -m unittest tests/test_governance_current_state_lock.py`
- `long_regression` ->
  - `cargo test --tests --features long-tests -q`
  - `cargo test --test opt_p2_stable_regression_matrix -q` (smoke)
  - `cargo test --test bench_recall_gated_baseline -q` (smoke)

## 用法

```bash
# 直接指定 profile
scripts/gate_profile_runner.sh --profile full_regression

# 从 RESULT 文件读取 gate_profile
scripts/gate_profile_runner.sh --from-result memory/PLAN_RESULT.json

# 仅打印本次会执行的命令
scripts/gate_profile_runner.sh --from-result memory/VERIFY_RESULT.json --print-checks
```

`memory/*_RESULT.json` 的 `gate_profile` 字段与上表是一一映射关系。
