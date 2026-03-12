#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

FIXTURE_NAME="${1:-sift-128-euclidean}"
FIXTURE_URL="${2:-http://ann-benchmarks.com/${FIXTURE_NAME}.hdf5}"

ensure_local_command ssh
load_remote_config
require_remote_config REMOTE_HOST REMOTE_USER REMOTE_NATIVE_REPO_DIR REMOTE_NATIVE_LOG_DIR

run_remote_script "${REMOTE_NATIVE_REPO_DIR}" "${REMOTE_NATIVE_LOG_DIR}" "${FIXTURE_NAME}" "${FIXTURE_URL}" <<'EOF'
set -euo pipefail

src="$1"
log_dir="$2"
fixture_name="$3"
fixture_url="$4"
fixture_path="${src}/${fixture_name}.hdf5"
log_file="${log_dir}/fixture_${fixture_name}_$(date -u +%Y%m%dT%H%M%SZ).log"
mkdir -p "${log_dir}"

{
  echo "[fixture] src=${src}"
  echo "[fixture] fixture_name=${fixture_name}"
  echo "[fixture] fixture_url=${fixture_url}"
  echo "[fixture] fixture_path=${fixture_path}"

  existing_size=0
  if [[ -f "${fixture_path}" ]]; then
    existing_size=$(stat -c '%s' "${fixture_path}" 2>/dev/null || stat -f '%z' "${fixture_path}")
  fi
  echo "[fixture] existing_size=${existing_size}"

  if command -v curl >/dev/null 2>&1; then
    if [[ -f "${fixture_path}" && "${existing_size}" -gt 0 ]]; then
      curl -L --fail -C - --output "${fixture_path}" "${fixture_url}"
    else
      curl -L --fail --output "${fixture_path}" "${fixture_url}"
    fi
  elif command -v wget >/dev/null 2>&1; then
    if [[ -f "${fixture_path}" && "${existing_size}" -gt 0 ]]; then
      wget -c -O "${fixture_path}" "${fixture_url}"
    else
      wget -O "${fixture_path}" "${fixture_url}"
    fi
  else
    echo "missing curl/wget on remote host" >&2
    exit 127
  fi

  python3 - "${fixture_path}" <<'PY'
import pathlib, sys
path = pathlib.Path(sys.argv[1])
with path.open('rb') as f:
    magic = f.read(8)
print(f"magic={magic!r}")
if magic != b'\x89HDF\r\n\x1a\n':
    raise SystemExit(f"unexpected hdf5 magic for {path}: {magic!r}")
print(f"size_bytes={path.stat().st_size}")
PY
} | tee "${log_file}"

printf 'fixture_path=%s\n' "${fixture_path}"
printf 'log=%s\n' "${log_file}"
EOF
