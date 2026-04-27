#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

KNOWHERE_REF=""
FORCE_CLEAN="false"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ref)
            KNOWHERE_REF="$2"
            shift 2
            ;;
        --force-clean)
            FORCE_CLEAN="true"
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
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_NATIVE_REPO_DIR REMOTE_NATIVE_LOG_DIR

KNOWHERE_REF="${KNOWHERE_REF:-${HANNS_OFFICIAL_KNOWHERE_REF:-}}"
if [[ -z "${KNOWHERE_REF}" ]]; then
    echo "missing official Knowhere ref: pass --ref or set HANNS_OFFICIAL_KNOWHERE_REF" >&2
    exit 2
fi

case "${REMOTE_NATIVE_REPO_URL}" in
    https://github.com/zilliztech/knowhere|https://github.com/zilliztech/knowhere.git|git@github.com:zilliztech/knowhere.git)
        ;;
    *)
        echo "official Knowhere baseline must be zilliztech/knowhere, got ${REMOTE_NATIVE_REPO_URL}" >&2
        exit 2
        ;;
esac

run_remote_script "${REMOTE_NATIVE_REPO_DIR}" "${REMOTE_NATIVE_LOG_DIR}" "${REMOTE_NATIVE_REPO_URL}" "${KNOWHERE_REF}" "${FORCE_CLEAN}" <<'EOF'
set -euo pipefail

repo_dir="$1"
log_dir="$2"
repo_url="$3"
knowhere_ref="$4"
force_clean="$5"

mkdir -p "${log_dir}" "$(dirname "${repo_dir}")"
log_file="${log_dir}/official_knowhere_provenance_$(date -u +%Y%m%dT%H%M%SZ).log"
json_file="${log_file%.log}.json"

canonical_url() {
    case "$1" in
        https://github.com/zilliztech/knowhere|https://github.com/zilliztech/knowhere.git|git@github.com:zilliztech/knowhere.git)
            printf '%s\n' "https://github.com/zilliztech/knowhere"
            ;;
        *)
            printf '%s\n' "$1"
            ;;
    esac
}

{
    echo "[official-knowhere] repo_dir=${repo_dir}"
    echo "[official-knowhere] repo_url=${repo_url}"
    echo "[official-knowhere] ref=${knowhere_ref}"

    if [[ ! -d "${repo_dir}/.git" ]]; then
        rm -rf "${repo_dir}"
        git clone "${repo_url}" "${repo_dir}"
    fi

    origin="$(git -C "${repo_dir}" remote get-url origin)"
    origin_canonical="$(canonical_url "${origin}")"
    if [[ "${origin_canonical}" != "https://github.com/zilliztech/knowhere" ]]; then
        echo "non-Zilliz Knowhere origin is not allowed: ${origin}" >&2
        exit 2
    fi

    dirty_before="$(git -C "${repo_dir}" status --porcelain)"
    if [[ -n "${dirty_before}" && "${force_clean}" != "true" ]]; then
        echo "Knowhere worktree is dirty; rerun with --force-clean if this isolated remote dir may be reset" >&2
        git -C "${repo_dir}" status --short >&2
        exit 2
    fi
    if [[ -n "${dirty_before}" ]]; then
        git -C "${repo_dir}" reset --hard HEAD
        git -C "${repo_dir}" clean -fdx
    fi

    git -C "${repo_dir}" fetch origin "${knowhere_ref}"
    git -C "${repo_dir}" checkout --detach FETCH_HEAD
    git -C "${repo_dir}" reset --hard HEAD
    git -C "${repo_dir}" clean -fdx

    dirty_after="$(git -C "${repo_dir}" status --porcelain)"
    if [[ -n "${dirty_after}" ]]; then
        echo "Knowhere worktree is dirty after reset" >&2
        git -C "${repo_dir}" status --short >&2
        exit 2
    fi

    commit="$(git -C "${repo_dir}" rev-parse HEAD)"
    hostname_value="$(hostname)"
    uname_value="$(uname -a)"
    if command -v lscpu >/dev/null 2>&1; then
        lscpu_text="$(lscpu)"
    else
        lscpu_text="lscpu unavailable"
    fi
    lscpu_hash="$(printf '%s' "${lscpu_text}" | sha256sum | awk '{print $1}')"

    python3 - "${json_file}" <<PY
import json
import pathlib
payload = {
    "authority": {
        "surface": "HannsDB-x86",
        "is_authority": True,
        "eligible_for_verdict": True,
        "runtime_proof": {
            "ssh_alias_or_host": "HannsDB-x86",
            "hostname": ${hostname_value@Q},
            "uname": ${uname_value@Q},
            "lscpu_hash": ${lscpu_hash@Q},
            "remote_log_path": ${log_file@Q},
            "wrapper_script": "scripts/remote/official_knowhere_provenance.sh",
        },
    },
    "source": {
        "repo_url": "https://github.com/zilliztech/knowhere",
        "commit": ${commit@Q},
        "ref": ${knowhere_ref@Q},
        "dirty": False,
    },
}
pathlib.Path(${json_file@Q}).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\\n")
print(json.dumps(payload, sort_keys=True))
PY
} 2>&1 | tee "${log_file}"

printf 'provenance_json=%s\n' "${json_file}"
printf 'log=%s\n' "${log_file}"
EOF
