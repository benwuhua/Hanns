#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_ENV_FILE="${HOME}/.config/knowhere-x86-remote/remote.env"

expand_path() {
    local path="${1:-}"
    if [[ -z "${path}" ]]; then
        return 0
    fi
    if [[ "${path}" == "~" ]]; then
        printf '%s\n' "${HOME}"
    elif [[ "${path}" == ~/* ]]; then
        printf '%s/%s\n' "${HOME}" "${path#~/}"
    else
        printf '%s\n' "${path}"
    fi
}

ensure_local_command() {
    local cmd="${1}"
    if ! command -v "${cmd}" >/dev/null 2>&1; then
        echo "missing local command: ${cmd}" >&2
        exit 1
    fi
}

load_remote_config() {
    local env_file="${HANNS_REMOTE_ENV:-${DEFAULT_ENV_FILE}}"
    if [[ -f "${env_file}" ]]; then
        # shellcheck disable=SC1090
        source "${env_file}"
    fi

    REMOTE_HOST="${HANNS_REMOTE_HOST:-${REMOTE_HOST:-}}"
    REMOTE_USER="${HANNS_REMOTE_USER:-${REMOTE_USER:-root}}"
    REMOTE_PORT="${HANNS_REMOTE_PORT:-${REMOTE_PORT:-22}}"
    REMOTE_WORK_ROOT="${HANNS_REMOTE_WORK_ROOT:-${REMOTE_WORK_ROOT:-/data/work}}"
    # Do not inherit generic repo/build/log directories from the shared knowhere/PiPNN remote env.
    # hanns must always use its own isolated remote workspace unless explicitly overridden
    # with HANNS_* variables.
    REMOTE_REPO_DIR="${HANNS_REMOTE_REPO_DIR:-${REMOTE_WORK_ROOT}/hanns-src}"
    REMOTE_TARGET_DIR="${HANNS_REMOTE_TARGET_DIR:-${REMOTE_WORK_ROOT}/hanns-target}"
    REMOTE_LOG_DIR="${HANNS_REMOTE_LOG_DIR:-${REMOTE_WORK_ROOT}/hanns-logs}"
    REMOTE_NATIVE_REPO_DIR="${HANNS_REMOTE_NATIVE_REPO_DIR:-${REMOTE_WORK_ROOT}/knowhere-native-src}"
    REMOTE_NATIVE_BUILD_DIR="${HANNS_REMOTE_NATIVE_BUILD_DIR:-${REMOTE_WORK_ROOT}/knowhere-native-build-benchmark}"
    REMOTE_NATIVE_LOG_DIR="${HANNS_REMOTE_NATIVE_LOG_DIR:-${REMOTE_WORK_ROOT}/knowhere-native-logs}"
    DEFAULT_BRANCH="${HANNS_DEFAULT_BRANCH:-$(git -C "${REPO_ROOT}" branch --show-current 2>/dev/null || printf 'main')}"
    DEFAULT_BUILD_TYPE="${HANNS_DEFAULT_BUILD_TYPE:-Release}"
    REMOTE_REPO_URL="${HANNS_REMOTE_REPO_URL:-$(git -C "${REPO_ROOT}" remote get-url origin 2>/dev/null || true)}"
    REMOTE_NATIVE_REPO_URL="${HANNS_REMOTE_NATIVE_REPO_URL:-https://github.com/zilliztech/knowhere.git}"
    REMOTE_NATIVE_DEFAULT_BRANCH="${HANNS_REMOTE_NATIVE_DEFAULT_BRANCH:-${DEFAULT_BRANCH:-main}}"
    REMOTE_CARGO_ENV_FILE="${HANNS_REMOTE_CARGO_ENV_FILE:-${REMOTE_CARGO_ENV_FILE:-\$HOME/.cargo/env}}"
    REMOTE_RUSTUP_TOOLCHAIN="${HANNS_REMOTE_RUSTUP_TOOLCHAIN:-${REMOTE_RUSTUP_TOOLCHAIN:-}}"
    SSH_IDENTITY_FILE="$(expand_path "${HANNS_SSH_IDENTITY_FILE:-${SSH_IDENTITY_FILE:-}}")"

    # Native baseline must never point back to the hanns repository.
    if [[ -n "${REMOTE_REPO_URL:-}" ]] && [[ "${REMOTE_NATIVE_REPO_URL}" == "${REMOTE_REPO_URL}" ]]; then
        echo "invalid remote native repo url: native baseline must use official knowhere, not hanns" >&2
        exit 1
    fi
}

require_remote_config() {
    local missing=0
    local var
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            echo "missing config: ${var}" >&2
            missing=1
        fi
    done
    if [[ "${missing}" -ne 0 ]]; then
        echo "load a config file via HANNS_REMOTE_ENV or ${DEFAULT_ENV_FILE}" >&2
        exit 1
    fi
}

remote_target() {
    printf '%s@%s' "${REMOTE_USER}" "${REMOTE_HOST}"
}

ssh_base_args() {
    SSH_BASE_ARGS=(-o StrictHostKeyChecking=accept-new -p "${REMOTE_PORT}")
    if [[ -n "${SSH_IDENTITY_FILE}" ]]; then
        SSH_BASE_ARGS+=(-i "${SSH_IDENTITY_FILE}" -o IdentitiesOnly=yes)
    fi
}

run_ssh() {
    local target
    target="$(remote_target)"
    ssh_base_args
    ssh "${SSH_BASE_ARGS[@]}" "${target}" "$@"
}

run_remote_script() {
    local target
    local remote_cmd
    local arg
    target="$(remote_target)"
    ssh_base_args
    remote_cmd="bash -s --"
    for arg in "$@"; do
        remote_cmd+=" $(printf '%q' "${arg}")"
    done
    ssh "${SSH_BASE_ARGS[@]}" "${target}" "${remote_cmd}"
}

run_rsync() {
    ssh_base_args
    rsync -az -e "ssh ${SSH_BASE_ARGS[*]}" "$@"
}

timestamp_utc() {
    date -u +"%Y%m%dT%H%M%SZ"
}

print_config_summary() {
    cat <<EOF
remote_host=${REMOTE_HOST}
remote_user=${REMOTE_USER}
remote_port=${REMOTE_PORT}
remote_work_root=${REMOTE_WORK_ROOT}
remote_repo_dir=${REMOTE_REPO_DIR}
remote_target_dir=${REMOTE_TARGET_DIR}
remote_log_dir=${REMOTE_LOG_DIR}
remote_native_repo_dir=${REMOTE_NATIVE_REPO_DIR}
remote_native_build_dir=${REMOTE_NATIVE_BUILD_DIR}
remote_native_log_dir=${REMOTE_NATIVE_LOG_DIR}
default_branch=${DEFAULT_BRANCH}
default_build_type=${DEFAULT_BUILD_TYPE}
remote_repo_url=${REMOTE_REPO_URL:-<unset>}
remote_native_repo_url=${REMOTE_NATIVE_REPO_URL:-<unset>}
remote_native_default_branch=${REMOTE_NATIVE_DEFAULT_BRANCH}
remote_cargo_env_file=${REMOTE_CARGO_ENV_FILE}
remote_rustup_toolchain=${REMOTE_RUSTUP_TOOLCHAIN:-<default>}
ssh_identity_file=${SSH_IDENTITY_FILE:-<default>}
EOF
}
