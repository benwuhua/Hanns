#!/usr/bin/env bash
set -euo pipefail

# Unified gate profile runner for plan/dev/verify stages.
# Usage examples:
#   scripts/gate_profile_runner.sh --profile default_regression
#   scripts/gate_profile_runner.sh --from-result memory/PLAN_RESULT.json --print-checks
#   scripts/gate_profile_runner.sh --profile long_regression --dry-run

PROFILE=""
FROM_RESULT=""
PRINT_CHECKS=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --from-result)
      FROM_RESULT="${2:-}"
      shift 2
      ;;
    --print-checks)
      PRINT_CHECKS=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      cat <<'USAGE'
Usage:
  gate_profile_runner.sh [--profile <name>] [--from-result <json>] [--print-checks] [--dry-run]

Profiles:
  default_regression -> cargo test -q
  full_regression    -> cargo test --lib -q; cargo test --tests -q; cargo test --doc -q
  long_regression    -> cargo test --tests --features long-tests -q; plus long-tests smoke subset

Notes:
  - --from-result reads gate_profile from a RESULT json (PLAN/DEV/VERIFY)
  - --print-checks only prints resolved commands, no execution
  - --dry-run prints + resolved commands and exits
USAGE
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "$FROM_RESULT" ]]; then
  if [[ ! -f "$FROM_RESULT" ]]; then
    echo "result file not found: $FROM_RESULT" >&2
    exit 2
  fi
  PROFILE_FROM_JSON="$(python3 - "$FROM_RESULT" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)
print((data.get('gate_profile') or '').strip())
PY
)"
  if [[ -n "$PROFILE_FROM_JSON" ]]; then
    PROFILE="$PROFILE_FROM_JSON"
  fi
fi

if [[ -z "$PROFILE" ]]; then
  echo "missing profile: use --profile or --from-result" >&2
  exit 2
fi

COMMANDS=()
case "$PROFILE" in
  default_regression)
    COMMANDS=(
      "cargo test -q"
    )
    ;;
  full_regression)
    COMMANDS=(
      "cargo test --lib -q"
      "cargo test --tests -q"
      "cargo test --doc -q"
    )
    ;;
  long_regression)
    COMMANDS=(
      "cargo test --tests --features long-tests -q"
      "cargo test --test opt_p2_stable_regression_matrix -q"
      "cargo test --test bench_recall_gated_baseline -q"
    )
    ;;
  *)
    echo "unsupported gate profile: $PROFILE" >&2
    exit 2
    ;;
esac

if [[ "$PRINT_CHECKS" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
  printf '%s\n' "profile=$PROFILE"
  for cmd in "${COMMANDS[@]}"; do
    printf '%s\n' "$cmd"
  done
fi

if [[ "$PRINT_CHECKS" -eq 1 || "$DRY_RUN" -eq 1 ]]; then
  exit 0
fi

for cmd in "${COMMANDS[@]}"; do
  echo "[gate_profile_runner] $cmd"
  bash -lc "$cmd"
done

echo "[gate_profile_runner] done profile=$PROFILE"