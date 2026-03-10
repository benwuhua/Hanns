#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/remote/common.sh"

ensure_local_command ssh
ensure_local_command rsync

load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_TARGET_DIR REMOTE_LOG_DIR

echo "=== knowhere-rs remote bootstrap ==="
print_config_summary

echo "=== syncing workspace to remote authority ==="
bash "${REPO_ROOT}/scripts/remote/sync.sh" --mode rsync

echo "=== probing remote toolchain ==="
run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_CARGO_ENV_FILE}" "${REMOTE_RUSTUP_TOOLCHAIN}" <<'EOF'
set -euo pipefail

repo_dir="${1:-}"
cargo_env_file="${2:-$HOME/.cargo/env}"
rustup_toolchain="${3:-}"

if [[ -f "${cargo_env_file}" ]]; then
  # shellcheck disable=SC1090
  source "${cargo_env_file}"
fi

if [[ -n "${rustup_toolchain}" ]]; then
  export RUSTUP_TOOLCHAIN="${rustup_toolchain}"
fi

cd "${repo_dir}"
echo "remote_pwd=$(pwd)"
echo "remote_commit=$(git rev-parse HEAD)"
echo "remote_branch=$(git rev-parse --abbrev-ref HEAD)"
cargo --version
rustc --version
EOF

echo "=== remote authority ready ==="
