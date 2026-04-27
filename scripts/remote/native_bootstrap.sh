#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_NATIVE_REPO_DIR REMOTE_NATIVE_LOG_DIR

WITH_DISKANN="${HANNS_NATIVE_WITH_DISKANN:-ON}"
REMOTE_NATIVE_CONAN_HOME="${HANNS_REMOTE_NATIVE_CONAN_HOME:-${REMOTE_WORK_ROOT}/knowhere-native-conan2}"
REMOTE_NATIVE_TMPDIR="${HANNS_REMOTE_NATIVE_TMPDIR:-${REMOTE_WORK_ROOT}/knowhere-native-tmp}"
case "${WITH_DISKANN}" in
  ON|on|true|True|1)
    CONAN_WITH_DISKANN="True"
    ;;
  OFF|off|false|False|0)
    CONAN_WITH_DISKANN="False"
    ;;
  *)
    echo "invalid HANNS_NATIVE_WITH_DISKANN: ${WITH_DISKANN}" >&2
    exit 2
    ;;
esac

run_remote_script "${REMOTE_REPO_DIR}" "${REMOTE_NATIVE_REPO_DIR}" "${REMOTE_NATIVE_LOG_DIR}" "${CONAN_WITH_DISKANN}" "${REMOTE_NATIVE_CONAN_HOME}" "${REMOTE_NATIVE_TMPDIR}" <<'EOF'
set -euo pipefail
repo_root="$1"
src="$2"
log_dir="$3"
with_diskann="$4"
conan_home="$5"
tmp_dir="$6"
cd "${repo_root}"
bash scripts/remote/native_bootstrap_inner.sh "${src}" "${log_dir}" "${with_diskann}" "${conan_home}" "${tmp_dir}"
EOF
