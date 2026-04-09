#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${REPO_ROOT}"

source "${REPO_ROOT}/scripts/remote/common.sh"

ensure_local_command ssh
ensure_local_command rsync

load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_TARGET_DIR REMOTE_LOG_DIR

SYNC_CMD_OVERRIDE="${KNOWHERE_RS_INIT_SYNC_CMD:-}"
PROBE_CMD_OVERRIDE="${KNOWHERE_RS_INIT_PROBE_CMD:-}"

echo "=== knowhere-rs remote bootstrap ==="
print_config_summary

echo "=== syncing workspace to remote authority ==="
if [[ -n "${SYNC_CMD_OVERRIDE}" ]]; then
    bash -lc "${SYNC_CMD_OVERRIDE}"
else
    bash "${REPO_ROOT}/scripts/remote/sync.sh" --mode rsync
fi

echo "=== probing remote toolchain ==="
if [[ -n "${PROBE_CMD_OVERRIDE}" ]]; then
    bash -lc "${PROBE_CMD_OVERRIDE}"
else
run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_CARGO_ENV_FILE}" "${REMOTE_RUSTUP_TOOLCHAIN}" <<'EOF'
set -euo pipefail

repo_dir="${1:-}"
cargo_env_file="${2:-$HOME/.cargo/env}"
rustup_toolchain="${3:-}"

source "${repo_dir}/scripts/remote/remote_env.sh"
load_remote_cargo_env "${cargo_env_file}" "${rustup_toolchain}"
ensure_remote_rust_components rustfmt clippy

cd "${repo_dir}"
echo "remote_pwd=$(pwd)"
echo "remote_commit=$(git rev-parse HEAD)"
echo "remote_branch=$(git rev-parse --abbrev-ref HEAD)"
cargo --version
rustc --version
EOF
fi

echo "=== remote authority ready ==="
