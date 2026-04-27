#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

usage() {
    cat <<'EOF'
Usage:
  remote_background.sh start --command '<shell-command>' [--repo-dir <dir>] [--log-dir <dir>]
  remote_background.sh status --status-file <remote-path>
  remote_background.sh wait --status-file <remote-path> [--poll-interval <seconds>]
  remote_background.sh tail --log <remote-path> [--lines <n>]
EOF
}

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
    usage
    exit 1
fi
shift || true

load_remote_config
REPO_DIR="${REMOTE_REPO_DIR}"
LOG_DIR="${REMOTE_LOG_DIR}"
COMMAND=""
STATUS_FILE=""
LOG_FILE=""
LINES="80"
POLL_INTERVAL_SECONDS="5"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --repo-dir)
            REPO_DIR="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --command)
            COMMAND="$2"
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
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_REPO_DIR REMOTE_LOG_DIR

case "${MODE}" in
    start)
        if [[ -z "${COMMAND}" ]]; then
            echo "start mode requires --command" >&2
            exit 1
        fi
        RUN_ID="$(timestamp_utc)_$$"
        REMOTE_METADATA_RAW="$(run_remote_script "${REPO_DIR}" "${LOG_DIR}" "${RUN_ID}" "${COMMAND}" <<'EOF'
set -euo pipefail

repo_dir="$1"
log_dir="$2"
run_id="$3"
command_override="$4"

mkdir -p "${log_dir}"
log_file="${log_dir}/remote_background_${run_id}.log"
status_file="${log_dir}/remote_background_${run_id}.status"

nohup env \
    RB_LOG_FILE="${log_file}" \
    RB_STATUS_FILE="${status_file}" \
    RB_RUN_ID="${run_id}" \
    RB_REPO_DIR="${repo_dir}" \
    RB_COMMAND="${command_override}" \
    bash -lc '
set -euo pipefail
printf "status=running\nrun_id=%s\nstarted_at=%s\nlog=%s\n" \
    "${RB_RUN_ID}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RB_LOG_FILE}" >"${RB_STATUS_FILE}"
{
    cd "${RB_REPO_DIR}"
    echo "[remote-background] run_id=${RB_RUN_ID}"
    echo "[remote-background] cwd=$(pwd)"
    echo "[remote-background] command=${RB_COMMAND}"
    echo "[remote-background] started_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    set +e
    bash -lc "${RB_COMMAND}"
    rc=$?
    set -e
    printf "[remote-background] exit_code=%s\n[remote-background] finished_at=%s\n" \
        "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >>"${RB_LOG_FILE}"
    printf "status=%s\nrun_id=%s\nexit_code=%s\nfinished_at=%s\nlog=%s\n" \
        "$([[ ${rc} -eq 0 ]] && printf ok || printf failed)" \
        "${RB_RUN_ID}" "${rc}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RB_LOG_FILE}" >"${RB_STATUS_FILE}"
    exit "${rc}"
} >"${RB_LOG_FILE}" 2>&1
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
