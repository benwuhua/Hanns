#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODE="git"
REF=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --ref)
            REF="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ensure_local_command ssh
ensure_local_command git
ensure_local_command rsync
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR

REF="${REF:-${DEFAULT_BRANCH}}"

case "${MODE}" in
    git)
        require_remote_config REMOTE_REPO_URL
        run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_REPO_URL}" "${REF}" <<'EOF'
set -euo pipefail
repo_dir="$1"
repo_url="$2"
ref="$3"

if [[ ! -d "${repo_dir}/.git" ]]; then
    rm -rf "${repo_dir}"
    git clone "${repo_url}" "${repo_dir}"
else
    git -C "${repo_dir}" reset --hard HEAD || true
    git -C "${repo_dir}" clean -fdx || true
fi

git -C "${repo_dir}" fetch --all --prune
if git -C "${repo_dir}" show-ref --verify --quiet "refs/remotes/origin/${ref}"; then
    git -C "${repo_dir}" checkout -B "${ref}" "origin/${ref}"
    git -C "${repo_dir}" reset --hard "origin/${ref}"
else
    git -C "${repo_dir}" checkout --detach "${ref}"
    git -C "${repo_dir}" reset --hard "${ref}"
fi
git -C "${repo_dir}" clean -fdx

printf 'sync_mode=git\n'
printf 'commit=%s\n' "$(git -C "${repo_dir}" rev-parse HEAD)"
printf 'branch=%s\n' "$(git -C "${repo_dir}" rev-parse --abbrev-ref HEAD)"
EOF
        ;;
    rsync)
        run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_TARGET_DIR}" "${REMOTE_LOG_DIR}" <<'EOF'
set -euo pipefail
mkdir -p "$1" "$2" "$3"
EOF
        run_rsync \
            --delete \
            --exclude=.git \
            --exclude=target \
            --exclude=.cache \
            --exclude=.cargo-home \
            --exclude=data \
            --exclude=cpp_bench_build \
            --exclude=benchmark_results/native_logs \
            --exclude=.tmp* \
            --exclude=memory/*.json \
            --exclude=memory/*.md \
            "${REPO_ROOT}/" "$(remote_target):${REMOTE_REPO_DIR}/"
        echo "sync_mode=rsync"
        ;;
    *)
        echo "unsupported mode: ${MODE}" >&2
        exit 1
        ;;
esac
