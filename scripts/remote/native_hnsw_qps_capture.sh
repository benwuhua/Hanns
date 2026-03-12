#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

load_remote_config
BUILD_DIR="${REMOTE_NATIVE_BUILD_DIR}"
LOG_DIR="${REMOTE_NATIVE_LOG_DIR}"
GTEST_FILTER="Benchmark_float_qps.TEST_HNSW"
REMOTE_BIN="${BUILD_DIR}/benchmark/benchmark_float_qps"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            BUILD_DIR="$2"
            REMOTE_BIN="${BUILD_DIR}/benchmark/benchmark_float_qps"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --gtest-filter)
            GTEST_FILTER="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ensure_local_command ssh
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_NATIVE_REPO_DIR REMOTE_NATIVE_BUILD_DIR REMOTE_NATIVE_LOG_DIR

run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_NATIVE_REPO_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${REMOTE_BIN}" "${GTEST_FILTER}" <<'EOF'
set -euo pipefail

repo_root="$1"
repo_dir="$2"
build_dir="$3"
log_dir="$4"
remote_bin="$5"
gtest_filter="$6"

cd "${repo_dir}"

mkdir -p "${log_dir}"
log_file="${log_dir}/native_hnsw_qps_$(date -u +%Y%m%dT%H%M%SZ).log"

if [[ ! -x "${remote_bin}" ]]; then
    echo "missing benchmark binary: ${remote_bin}" >&2
    exit 127
fi

link_txt="${build_dir}/benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt"
if [[ ! -f "${link_txt}" ]]; then
    echo "missing link.txt for ${remote_bin}: ${link_txt}" >&2
    exit 127
fi

link_rpath="$(python3 - "${link_txt}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'-Wl,-rpath,([^ ]+)', text)
print(m.group(1) if m else '')
PY
)"
if [[ -z "${link_rpath}" ]]; then
    echo "failed to resolve runtime rpath from ${link_txt}" >&2
    exit 127
fi

export LD_LIBRARY_PATH="${link_rpath}:${build_dir}:${build_dir}/milvus-common-build:${LD_LIBRARY_PATH:-}"

set +e
"${remote_bin}" --gtest_filter="${gtest_filter}" --gtest_color=no >"${log_file}" 2>&1
rc=$?
set -e

if [[ ${rc} -ne 0 ]] && grep -q 'Metric name grpc.xds_client.resource_updates_valid has already been registered.' "${log_file}"; then
    patched_src="/data/work/knowhere-native-linkfix-src"
    patched_build="/data/work/knowhere-native-linkfix-build"
    linkfix_meta="$(bash "${repo_root}/scripts/remote/native_linkfix_remote.sh" \
      "${repo_root}" "${repo_dir}" "${patched_src}" "${patched_build}" "${log_dir}" "benchmark_float_qps")"
    patched_repo_dir="$(awk -F= '/^patched_repo_dir=/{print $2; exit}' <<<"${linkfix_meta}")"
    patched_build_dir="$(awk -F= '/^patched_build_dir=/{print $2; exit}' <<<"${linkfix_meta}")"
    patched_bin="$(awk -F= '/^patched_bin=/{print $2; exit}' <<<"${linkfix_meta}")"
    patched_link_txt="${patched_build_dir}/benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt"
    patched_link_rpath="$(python3 - "${patched_link_txt}" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'-Wl,-rpath,([^ ]+)', text)
print(m.group(1) if m else '')
PY
)"
    export LD_LIBRARY_PATH="${patched_link_rpath}:${patched_build_dir}:${patched_build_dir}/milvus-common-build:${LD_LIBRARY_PATH:-}"
    log_file="${log_dir}/native_hnsw_qps_linkfix_$(date -u +%Y%m%dT%H%M%SZ).log"
    set +e
    (cd "${patched_repo_dir}" && "${patched_bin}" --gtest_filter="${gtest_filter}" --gtest_color=no >"${log_file}" 2>&1)
    rc=$?
    set -e
    printf 'fallback=linkfix_worktree\n'
    printf 'patched_repo_dir=%s\n' "${patched_repo_dir}"
    printf 'patched_build_dir=%s\n' "${patched_build_dir}"
    printf 'patched_bin=%s\n' "${patched_bin}"
fi

printf 'log=%s\n' "${log_file}"
printf 'exit_code=%s\n' "${rc}"

if [[ ${rc} -ne 0 ]]; then
    echo "native benchmark exited with rc=${rc}; inspect ${log_file}" >&2
    exit "${rc}"
fi
EOF
