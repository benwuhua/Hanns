#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  native_benchmark_background.sh start [--repo-dir <dir>] [--build-dir <dir>] [--log-dir <dir>] --gtest-filter <filter>
  native_benchmark_background.sh status --status-file <remote-path>
  native_benchmark_background.sh wait --status-file <remote-path> [--poll-interval <seconds>]
  native_benchmark_background.sh tail --log <remote-path> [--lines <n>]
EOF
}

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
    usage
    exit 1
fi
shift || true

load_remote_config
REPO_DIR="${REMOTE_NATIVE_REPO_DIR}"
BUILD_DIR="${REMOTE_NATIVE_BUILD_DIR}"
LOG_DIR="${REMOTE_NATIVE_LOG_DIR}"
GTEST_FILTER=""
STATUS_FILE=""
LOG_FILE=""
LINES="80"
POLL_INTERVAL_SECONDS="5"
REMOTE_BIN="${BUILD_DIR}/benchmark/benchmark_float_qps"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir)
            REPO_DIR="$2"
            shift 2
            ;;
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
        --status-file)
            STATUS_FILE="$2"
            shift 2
            ;;
        --log)
            LOG_FILE="$2"
            shift 2
            ;;
        --lines)
            LINES="$2"
            shift 2
            ;;
        --poll-interval)
            POLL_INTERVAL_SECONDS="$2"
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

case "${MODE}" in
    start)
        if [[ -z "${GTEST_FILTER}" ]]; then
            echo "start mode requires --gtest-filter" >&2
            exit 1
        fi

        RUN_ID="$(timestamp_utc)_$$"
        REMOTE_METADATA_RAW="$(run_remote_script "${REMOTE_REPO_DIR}" "${REPO_DIR}" "${BUILD_DIR}" "${LOG_DIR}" "${REMOTE_BIN}" "${GTEST_FILTER}" "${RUN_ID}" <<'EOF'
set -euo pipefail

repo_root="$1"
repo_dir="$2"
build_dir="$3"
log_dir="$4"
remote_bin="$5"
gtest_filter="$6"
run_id="$7"

cd "${repo_dir}"
mkdir -p "${log_dir}"

log_file="${log_dir}/native_benchmark_${run_id}.log"
status_file="${log_dir}/native_benchmark_${run_id}.status"

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

nohup env \
    NATIVE_BENCH_LOG_FILE="${log_file}" \
    NATIVE_BENCH_STATUS_FILE="${status_file}" \
    NATIVE_BENCH_RUN_ID="${run_id}" \
    NATIVE_BENCH_REPO_ROOT="${repo_root}" \
    NATIVE_BENCH_REPO_DIR="${repo_dir}" \
    NATIVE_BENCH_BUILD_DIR="${build_dir}" \
    NATIVE_BENCH_LOG_DIR="${log_dir}" \
    NATIVE_BENCH_BIN="${remote_bin}" \
    NATIVE_BENCH_FILTER="${gtest_filter}" \
    NATIVE_BENCH_LINK_RPATH="${link_rpath}" \
    bash -lc '
set -euo pipefail

printf "status=running\nrun_id=%s\nstarted_at=%s\nlog=%s\nfilter=%s\n" \
    "${NATIVE_BENCH_RUN_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${NATIVE_BENCH_LOG_FILE}" "${NATIVE_BENCH_FILTER}" \
    >"${NATIVE_BENCH_STATUS_FILE}"

export LD_LIBRARY_PATH="${NATIVE_BENCH_LINK_RPATH}:${NATIVE_BENCH_BUILD_DIR}:${NATIVE_BENCH_BUILD_DIR}/milvus-common-build:${LD_LIBRARY_PATH:-}"

{
    cd "${NATIVE_BENCH_REPO_DIR}"
    echo "[native-benchmark] run_id=${NATIVE_BENCH_RUN_ID}"
    echo "[native-benchmark] cwd=$(pwd)"
    echo "[native-benchmark] filter=${NATIVE_BENCH_FILTER}"
    echo "[native-benchmark] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

    set +e
    "${NATIVE_BENCH_BIN}" --gtest_filter="${NATIVE_BENCH_FILTER}" --gtest_color=no
    rc=$?
    set -e

    if [[ ${rc} -ne 0 ]] && grep -q "Metric name grpc.xds_client.resource_updates_valid has already been registered." "${NATIVE_BENCH_LOG_FILE}"; then
        patched_src="/data/work/knowhere-native-linkfix-src"
        patched_build="/data/work/knowhere-native-linkfix-build"
        linkfix_meta="$(bash "${NATIVE_BENCH_REPO_ROOT}/scripts/remote/native_linkfix_remote.sh" \
            "${NATIVE_BENCH_REPO_ROOT}" "${NATIVE_BENCH_REPO_DIR}" "${patched_src}" "${patched_build}" "${NATIVE_BENCH_LOG_DIR}" "benchmark_float_qps")"
        patched_repo_dir="$(awk -F= "/^patched_repo_dir=/{print \$2; exit}" <<<"${linkfix_meta}")"
        patched_build_dir="$(awk -F= "/^patched_build_dir=/{print \$2; exit}" <<<"${linkfix_meta}")"
        patched_bin="$(awk -F= "/^patched_bin=/{print \$2; exit}" <<<"${linkfix_meta}")"
        patched_link_txt="${patched_build_dir}/benchmark/CMakeFiles/benchmark_float_qps.dir/link.txt"
        patched_link_rpath="$(python3 - "${patched_link_txt}" <<PY
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r"-Wl,-rpath,([^ ]+)", text)
print(m.group(1) if m else "")
PY
)"
        export LD_LIBRARY_PATH="${patched_link_rpath}:${patched_build_dir}:${patched_build_dir}/milvus-common-build:${LD_LIBRARY_PATH:-}"
        fallback_log="${NATIVE_BENCH_LOG_DIR}/native_benchmark_linkfix_${NATIVE_BENCH_RUN_ID}.log"
        set +e
        (cd "${patched_repo_dir}" && "${patched_bin}" --gtest_filter="${NATIVE_BENCH_FILTER}" --gtest_color=no >"${fallback_log}" 2>&1)
        rc=$?
        set -e
        if [[ ${rc} -eq 0 ]]; then
            cp "${fallback_log}" "${NATIVE_BENCH_LOG_FILE}"
        fi
    fi

    printf "[native-benchmark] exit_code=%s\n[native-benchmark] finished_at=%s\n" \
        "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${NATIVE_BENCH_LOG_FILE}"
    printf "status=%s\nrun_id=%s\nexit_code=%s\nfinished_at=%s\nlog=%s\nfilter=%s\n" \
        "$([[ ${rc} -eq 0 ]] && printf ok || printf failed)" \
        "${NATIVE_BENCH_RUN_ID}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${NATIVE_BENCH_LOG_FILE}" "${NATIVE_BENCH_FILTER}" >"${NATIVE_BENCH_STATUS_FILE}"
    exit "${rc}"
} >"${NATIVE_BENCH_LOG_FILE}" 2>&1
' >/dev/null 2>&1 &
pid=$!

printf 'pid=%s\n' "${pid}"
printf 'log=%s\n' "${log_file}"
printf 'status_file=%s\n' "${status_file}"
printf 'run_id=%s\n' "${run_id}"
EOF
)"
        printf '%s\n' "${REMOTE_METADATA_RAW}"
        ;;
    status)
        if [[ -z "${STATUS_FILE}" ]]; then
            echo "status mode requires --status-file" >&2
            exit 1
        fi
        run_ssh "if [[ -f $(printf '%q' "${STATUS_FILE}") ]]; then cat $(printf '%q' "${STATUS_FILE}"); else echo missing; fi"
        ;;
    wait)
        if [[ -z "${STATUS_FILE}" ]]; then
            echo "wait mode requires --status-file" >&2
            exit 1
        fi
        while true; do
            STATUS_CONTENT="$(run_ssh "if [[ -f $(printf '%q' "${STATUS_FILE}") ]]; then cat $(printf '%q' "${STATUS_FILE}"); else echo missing; fi")"
            printf '%s\n' "${STATUS_CONTENT}"
            if grep -q '^status=ok$' <<<"${STATUS_CONTENT}"; then
                exit 0
            fi
            if grep -q '^status=failed$' <<<"${STATUS_CONTENT}"; then
                exit_code="$(awk -F= '/^exit_code=/{print $2; exit}' <<<"${STATUS_CONTENT}")"
                exit "${exit_code:-1}"
            fi
            if grep -q '^status=conflict$' <<<"${STATUS_CONTENT}"; then
                exit 91
            fi
            sleep "${POLL_INTERVAL_SECONDS}"
        done
        ;;
    tail)
        if [[ -z "${LOG_FILE}" ]]; then
            echo "tail mode requires --log" >&2
            exit 1
        fi
        run_ssh "if [[ -f $(printf '%q' "${LOG_FILE}") ]]; then tail -n $(printf '%q' "${LINES}") $(printf '%q' "${LOG_FILE}"); else echo missing; fi"
        ;;
    *)
        usage
        exit 1
        ;;
esac
