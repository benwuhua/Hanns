#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

PROFILE=""
FROM_RESULT=""
COMMAND=""
POLL_INTERVAL_SECONDS="5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --from-result)
            FROM_RESULT="$2"
            shift 2
            ;;
        --command)
            COMMAND="$2"
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
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_TARGET_DIR REMOTE_LOG_DIR

RUN_ID="$(timestamp_utc)_$$"

REMOTE_METADATA_RAW="$(run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_TARGET_DIR}" "${REMOTE_LOG_DIR}" "${REMOTE_CARGO_ENV_FILE}" "${REMOTE_RUSTUP_TOOLCHAIN}" "${RUN_ID}" "${PROFILE}" "${FROM_RESULT}" "${COMMAND}" <<'EOF'
set -euo pipefail

repo_dir="${1:-}"
target_dir="${2:-}"
log_dir="${3:-}"
cargo_env_file="${4:-$HOME/.cargo/env}"
rustup_toolchain="${5:-}"
run_id="${6:-manual}"
profile="${7:-}"
from_result="${8:-}"
command_override="${9:-}"

source "${repo_dir}/scripts/remote/remote_env.sh"
load_remote_cargo_env "${cargo_env_file}" "${rustup_toolchain}"
ensure_remote_rust_components rustfmt clippy
export CARGO_TARGET_DIR="${target_dir}"

mkdir -p "${log_dir}"
log_file="${log_dir}/test_${run_id}.log"
status_file="${log_dir}/test_${run_id}.status"
lock_file="${log_dir}/hanns-test.lock"

if [[ -n "${command_override}" ]]; then
    test_cmd="${command_override}"
elif [[ -n "${from_result}" ]]; then
    test_cmd="scripts/gate_profile_runner.sh --from-result ${from_result}"
else
    test_cmd="scripts/gate_profile_runner.sh --profile ${profile}"
fi

nohup env \
    TEST_LOCK_FILE="${lock_file}" \
    TEST_LOG_FILE="${log_file}" \
    TEST_STATUS_FILE="${status_file}" \
    TEST_RUN_ID="${run_id}" \
    TEST_COMMAND="${test_cmd}" \
    TEST_REPO_DIR="${repo_dir}" \
    bash -lc '
set -euo pipefail
cleanup() {
    flock -u 9 || true
}
exec 9>"${TEST_LOCK_FILE}"
flock -n 9 || {
    printf "status=conflict\nrun_id=%s\nfinished_at=%s\nmessage=another hanns remote test is still active\n" \
        "${TEST_RUN_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${TEST_STATUS_FILE}"
    exit 91
}
trap cleanup EXIT
printf "status=running\nrun_id=%s\nstarted_at=%s\n" \
    "${TEST_RUN_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"${TEST_STATUS_FILE}"
{
    cd "${TEST_REPO_DIR}"
    git config --global --add safe.directory "${TEST_REPO_DIR}" || true
    if [[ ! -f Cargo.toml || ! -f src/lib.rs ]]; then
        echo "[test] invalid repo baseline: expected Cargo.toml and src/lib.rs under ${TEST_REPO_DIR}"
        exit 92
    fi
    if ! command -v cargo >/dev/null 2>&1; then
        echo "[test] missing cargo on remote PATH=${PATH}"
        exit 127
    fi
    echo "[test] run_id=${TEST_RUN_ID}"
    echo "[test] cwd=$(pwd)"
    echo "[test] command=${TEST_COMMAND}"
    echo "[test] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    runner_script="$(mktemp /tmp/hanns-test-run.XXXXXX.sh)"
    cat >"${runner_script}" <<RUNNER
#!/usr/bin/env bash
set -euo pipefail
cd "${TEST_REPO_DIR}"
eval "${TEST_COMMAND}"
RUNNER
    chmod +x "${runner_script}"
    set +e
    bash "${runner_script}"
    rc=$?
    rm -f "${runner_script}"
    set -e
    printf "status=%s\nrun_id=%s\nexit_code=%s\nfinished_at=%s\nlog=%s\n" \
        "$([[ ${rc} -eq 0 ]] && printf ok || printf failed)" \
        "${TEST_RUN_ID}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TEST_LOG_FILE}" >"${TEST_STATUS_FILE}"
    printf "[test] exit_code=%s\n[test] finished_at=%s\n" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${TEST_LOG_FILE}"
    exit "${rc}"
} >"${TEST_LOG_FILE}" 2>&1
' >/dev/null 2>&1 &
pid=$!
printf 'pid=%s\n' "${pid}"
printf 'log=%s\n' "${log_file}"
printf 'status_file=%s\n' "${status_file}"
EOF
)"

PID=""
LOG_FILE=""
STATUS_FILE=""
while IFS= read -r line; do
    case "${line}" in
        pid=*)
            PID="${line#pid=}"
            ;;
        log=*)
            LOG_FILE="${line#log=}"
            ;;
        status_file=*)
            STATUS_FILE="${line#status_file=}"
            ;;
    esac
done <<<"${REMOTE_METADATA_RAW}"

if [[ -z "${PID}" || -z "${LOG_FILE}" || -z "${STATUS_FILE}" ]]; then
    echo "failed to initialize remote test run" >&2
    exit 1
fi

while true; do
    STATUS_CONTENT="$(run_ssh "if [[ -f $(printf '%q' "${STATUS_FILE}") ]]; then cat $(printf '%q' "${STATUS_FILE}"); fi")"
    if grep -q '^status=ok$' <<<"${STATUS_CONTENT}"; then
        printf 'test=ok\n'
        printf 'pid=%s\n' "${PID}"
        printf 'log=%s\n' "${LOG_FILE}"
        printf 'status_file=%s\n' "${STATUS_FILE}"
        printf '%s\n' "${STATUS_CONTENT}"
        exit 0
    fi
    if grep -q '^status=failed$' <<<"${STATUS_CONTENT}"; then
        printf 'test=failed\n' >&2
        printf 'pid=%s\n' "${PID}" >&2
        printf 'log=%s\n' "${LOG_FILE}" >&2
        printf 'status_file=%s\n' "${STATUS_FILE}" >&2
        printf '%s\n' "${STATUS_CONTENT}" >&2
        exit_code="$(awk -F= '/^exit_code=/{print $2; exit}' <<<"${STATUS_CONTENT}")"
        exit "${exit_code:-1}"
    fi
    if grep -q '^status=conflict$' <<<"${STATUS_CONTENT}"; then
        printf 'test=conflict\n' >&2
        printf 'pid=%s\n' "${PID}" >&2
        printf 'log=%s\n' "${LOG_FILE}" >&2
        printf 'status_file=%s\n' "${STATUS_FILE}" >&2
        printf '%s\n' "${STATUS_CONTENT}" >&2
        exit 91
    fi
    sleep "${POLL_INTERVAL_SECONDS}"
done
