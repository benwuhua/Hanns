#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

BUILD_TYPE=""
ALL_TARGETS="true"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --no-all-targets)
            ALL_TARGETS="false"
            shift
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_TARGET_DIR REMOTE_LOG_DIR

BUILD_TYPE="${BUILD_TYPE:-${DEFAULT_BUILD_TYPE}}"

run_remote_script "${BUILD_TYPE}" "${ALL_TARGETS}" "${REMOTE_REPO_DIR}" "${REMOTE_TARGET_DIR}" "${REMOTE_LOG_DIR}" "${REMOTE_CARGO_ENV_FILE}" "${REMOTE_RUSTUP_TOOLCHAIN}" <<'EOF'
set -euo pipefail

build_type="${1:-Release}"
all_targets="${2:-true}"
repo_dir="${3:-}"
target_dir="${4:-}"
log_dir="${5:-}"
cargo_env_file="${6:-$HOME/.cargo/env}"
rustup_toolchain="${7:-}"

mkdir -p "${target_dir}" "${log_dir}"
log_file="${log_dir}/build_$(date -u +%Y%m%dT%H%M%SZ).log"

source "${repo_dir}/scripts/remote/remote_env.sh"
load_remote_cargo_env "${cargo_env_file}" "${rustup_toolchain}"
export CARGO_TARGET_DIR="${target_dir}"

build_args=()
if [[ "${build_type}" == "Release" ]]; then
    build_args+=(--release)
fi
if [[ "${all_targets}" == "true" ]]; then
    build_args+=(--all-targets)
fi

{
    echo "[build] repo_dir=${repo_dir}"
    echo "[build] target_dir=${target_dir}"
    echo "[build] build_type=${build_type}"
    git config --global --add safe.directory "${repo_dir}" || true
    echo "[build] commit=$(git -C "${repo_dir}" rev-parse HEAD)"
    cd "${repo_dir}"
    if ! command -v cargo >/dev/null 2>&1; then
        echo "[build] missing cargo on remote PATH=${PATH}" >&2
        exit 127
    fi
    cargo build "${build_args[@]}" --verbose
} 2>&1 | tee "${log_file}"

printf 'build=ok\n'
printf 'log=%s\n' "${log_file}"
EOF
